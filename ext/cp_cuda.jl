using TensorFactors
using TensorFactors: colnormalize!, flat_to_cp_factors, cp_factors_to_flat
using LinearAlgebra
using Tullio
using CUDA, KernelAbstractions
using Optim

function TensorFactors.cp_als(
    X::CuArray{T,3},
    cp_rank::Int;
    max_iter::Int=10000,
    dloss_rtol::Float64=1e-8,
    loss_rtol::Float64=1e-8,
    show_trace::Bool=false,
    show_every::Int=100,
) where {T<:Real}
    I, J, K = size(X)

    A = CUDA.randn(T, I, cp_rank)
    B = CUDA.randn(T, J, cp_rank)
    C = CUDA.randn(T, K, cp_rank)
    mttkrp_C = CuArray{Float64}(undef, K, cp_rank)
    V = CuArray{Float64}(undef, cp_rank, cp_rank)    # For Hardamard product of Gram matrices

    GtA = CuArray{Float64}(undef, cp_rank, cp_rank)
    GtB = CuArray{Float64}(undef, cp_rank, cp_rank)
    GtC = CuArray{Float64}(undef, cp_rank, cp_rank)

    C_loss = CuArray{Float64}(undef, K, cp_rank)  # Buffer for MTTKRP of C to evaluate loss
    GtC_loss = CuArray{Float64}(undef, cp_rank, cp_rank)  # Buffer for GtC to evaluate loss

    lambda = CUDA.ones(T, cp_rank)   # Normalization of factors to prevent numerical issues

    mul!(GtB, B', B)
    mul!(GtC, C', C)
    
    norm_tensor = norm(X)
    norm2_tensor = norm_tensor^2
    last_loss = sqrt(cp_loss((A, B, C), X, norm2_tensor)) / norm_tensor

    if show_trace
        println("Iteration 0: Time = 0.0 s, Loss = $last_loss")
    end
    start_time = time()
    for iter in 1:max_iter
        # Update A
        @tullio A[i, r] = X[i, j, k] * B[j, r] * C[k, r]
        @. V = GtB * GtC
        rdiv!(A, cholesky!(Symmetric(V)))  # Solve A * V_A = mttkrp_A
        colnormalize!(A)  # Absorb scaling into lambda to prevent numerical issues
        mul!(GtA, A', A)

        # Update B
        @tullio B[j, r] = X[i, j, k] * A[i, r] * C[k, r]
        @. V = GtA * GtC
        rdiv!(B, cholesky!(Symmetric(V)))
        colnormalize!(B)
        mul!(GtB, B', B)

        # Update C
        @tullio mttkrp_C[k, r] = X[i, j, k] * A[i, r] * B[j, r]
        @. V = GtA * GtB
        copyto!(C, mttkrp_C)
        rdiv!(C, cholesky!(Symmetric(V)))
        colnormalize!(C, lambda)
        mul!(GtC, C', C)

        # Evaluate loss
        @. C_loss = C * lambda'
        mul!(GtC_loss, C_loss', C_loss)
        loss = sqrt(cp_loss((GtA, GtB, GtC_loss), C_loss, mttkrp_C, norm2_tensor)) / norm_tensor

        if show_trace && iter % show_every == 0
            println("Iteration $iter: Time = $(time() - start_time) s, Loss = $loss")
        end

        stop_criterion = (abs(last_loss - loss) < dloss_rtol
                          || loss < loss_rtol)
        if iter > 1 && stop_criterion
            show_trace && println("Converged at iteration $iter: Time = $(time() - start_time) s, Loss = $loss")
            break
        end
        last_loss = loss
    end

    return lambda, A, B, C
end

@inline function TensorFactors.cp_loss(
    factors::NTuple{3, <:CuArray{T, 2}},
    X::CuArray{U, 3},
    X_norm2::V
) where {T <: Number, U <: Number, V <: Real}
    A, B, C = factors
    @tullio mttkrp_C[k, r] := X[i, j, k] * A[i, r] * B[j, r]
    Gt_A = A' * A
    Gt_B = B' * B
    Gt_C = C' * C
    loss = X_norm2 - 2.0 * dot(mttkrp_C, C) + sum(Gt_A .* Gt_B .* Gt_C)
    return max(loss, zero(T))
end

function TensorFactors.cp_loss(
    p::CuArray{T, 1},
    X::CuArray{U, 3},
    cp_rank::Int,
    norm2_X::V,
    GtA, GtB, GtC, H # Buffer
)::T where {T <: Real, U <: Real, V <: Real}
    A, B, C = flat_to_cp_factors(p, cp_rank, size(X))
    @tullio mttkrp_C[k, r] := X[i, j, k] * A[i, r] * B[j, r]
    mul!(GtA, A', A)
    mul!(GtB, B', B)
    mul!(GtC, C', C)
    @. H = GtA * GtB * GtC
    loss = norm2_X - 2 * dot(mttkrp_C, C) + sum(H)
    return max(loss, zero(T))
end

function TensorFactors.cp_loss_grad!(
    g::CuArray{T, 1},
    p::CuArray{T, 1},
    X::CuArray{U, 3},
    cp_rank::Int,
    GtA, GtB, GtC, H_A, H_B, H_C # Buffers
) where {T<: Real, U <: Real}
    A, B, C = flat_to_cp_factors(p, cp_rank, size(X))
    gA, gB, gC = flat_to_cp_factors(g, cp_rank, size(X))

    mul!(GtA, A', A)
    mul!(GtB, B', B)
    mul!(GtC, C', C)
    
    @. H_A = GtB * GtC
    @. H_B = GtA * GtC
    @. H_C = GtA * GtB

    @tullio gA[i, r] = X[i, j, k] * B[j, r] * C[k, r]
    @tullio gB[j, r] = X[i, j, k] * A[i, r] * C[k, r]
    @tullio gC[k, r] = X[i, j, k] * A[i, r] * B[j, r]

    mul!(gA, A, H_A, 2, -2)
    mul!(gB, B, H_B, 2, -2)
    mul!(gC, C, H_C, 2, -2)

    return nothing
end

function TensorFactors.cp_loss_fg!(
    G,
    p::CuArray{T, 1},
    X::CuArray{U, 3},
    cp_rank::Int,
    norm2_X::V,
    GtA, GtB, GtC, H_A, H_B, H_C # Buffers
) where {T <: Real, U <: Real, V <: Real}
    A, B, C = flat_to_cp_factors(p, cp_rank, size(X))

    mul!(GtA, A', A)
    mul!(GtB, B', B)
    mul!(GtC, C', C)
    
    @. H_A = GtB * GtC
    @. H_B = GtA * GtC
    @. H_C = GtA * GtB
    
    gA, gB, gC = flat_to_cp_factors(G, cp_rank, size(X))
    
    @tullio gA[i, r] = X[i, j, k] * B[j, r] * C[k, r]
    @tullio gB[j, r] = X[i, j, k] * A[i, r] * C[k, r]
    @tullio gC[k, r] = X[i, j, k] * A[i, r] * B[j, r]
    
    loss_val = zero(V)
    loss_val = norm2_X - 2.0 * dot(gC, C) + dot(GtA, H_A)
    loss_val = max(loss_val, zero(V))

    mul!(gA, A, H_A, 2, -2)
    mul!(gB, B, H_B, 2, -2)
    mul!(gC, C, H_C, 2, -2)
    
    return loss_val
end

function TensorFactors.cp_opt(
    method::Optim.AbstractOptimizer,
    X::CuArray{T, 3},
    cp_rank::Int;
    max_iter::Int = 10000,
    show_trace::Bool = false,
    show_every::Int = 100,
    init_factors::Union{Nothing, NTuple{3, CuArray{T, 2}}} = nothing,
) where {T <: Real}
    if init_factors === nothing
        p0 = CUDA.randn(T, cp_rank * sum(size(X)))
    else
        p0 = cp_factors_to_flat(init_factors)
    end

    norm2_X = sum(abs2, X)
    GtA, GtB, GtC, H_A, H_B, H_C = (CuArray{T, 2}(undef, cp_rank, cp_rank) for _ in 1:6) # Buffers

    f(u) = TensorFactors.cp_loss(u, X, cp_rank, norm2_X, GtA, GtB, GtC, H_A)
    g!(g, u) = TensorFactors.cp_loss_grad!(g, u, X, cp_rank, GtA, GtB, GtC, H_A, H_B, H_C)
    fg!(G, u) = TensorFactors.cp_loss_fg!(G, u, X, cp_rank, norm2_X, GtA, GtB, GtC, H_A, H_B, H_C)

    od = OnceDifferentiable(f, g!, fg!, p0)
    options = Optim.Options(iterations = max_iter, show_trace = show_trace, show_every = show_every)
    sol = optimize(od, p0, method, options)
    
    minimizer = Optim.minimizer(sol)

    cp_factors = flat_to_cp_factors(minimizer, cp_rank, size(X))
    return cp_factors
end

for N in 3:10
    # Generate independent variable names for each matrix, for example: :A1, :A2, ..., :AN
    vars = [Symbol("A", d) for d in 1:N]

    # Generate tensor index symbols, for example: :i1, :i2, ..., :iN
    idx = [Symbol("i", d) for d in 1:N]

    # Build the tuple unpacking expression on the left-hand side, for example: :(A1, A2, A3)
    unpack_tuple = Expr(:tuple, vars...)

    # Build the multiplication expression on the right-hand side, for example:
    # :( A1[i1, r] * A2[i2, r] * A3[i3, r] )
    rhs_args = [:( $(vars[d])[$(idx[d]), r] ) for d in 1:N]
    rhs = Expr(:call, :*, rhs_args...)

    # Generate and register the function
    @eval begin
        function TensorFactors.cp_contract(factors::NTuple{$N, <:CuArray{T, 2}}) where {T <: Number}
            $unpack_tuple = factors
            @tullio X[$(idx...)] := $rhs
            return X
        end
    end
end