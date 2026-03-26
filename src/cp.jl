using LinearAlgebra
using LoopVectorization

export cp_loss, cp_als

using LinearAlgebra

"""
    cp_loss(factors::NTuple{N, AbstractMatrix{T}}, X::AbstractArray{T, N}) where {N, T <: Real}

Computes the CP decomposition loss for an arbitrary-order tensor `X` from its factor
matrices `factors` using pure BLAS operations.

This method expands the squared Frobenius norm analytically to avoid explicit tensor
reconstruction and Khatri-Rao product formation. It sequentially contracts the tensor
with the factor vectors using matrix-vector multiplications, allowing the loss to be
evaluated efficiently with low memory overhead. This yields substantial speedups and
is well suited to high-performance and GPU-compatible tensor factorization workflows.

# Arguments
- `factors`: Tuple of `N` factor matrices defining the CP decomposition, where
  `factors[k]` has size `(size(X, k), R)` and all factor matrices share the same
  column dimension `R`.
- `X`: Input tensor of order `N`.

# Returns
- The squared Frobenius loss `||X - X̂||_F^2`, where `X̂` is the CP reconstruction
  induced by `factors`.
"""
function cp_loss(
    factors::NTuple{N, AbstractMatrix{T}}, 
    X::AbstractArray{T, N}
) where {N, T <: Number}
    R = size(factors[1], 2)
    norm2_X = norm(X)^2

    # ||X_recon||_F^2 = sum(*(A'A, B'B, C'C, ...))
    G = factors[1]' * factors[1]
    for n in 2:N
        G .*= factors[n]' * factors[n]
    end
    norm2_recon = sum(G)

    # Compute inner product <X, X_recon>
    # We contract X with the columns of the factor matrices (Tensor-Times-Vector)
    total_inner = zero(T)
    
    # Preallocate buffers for mode contractions to avoid allocations inside the loop.
    # buffer[n] holds the dynamically flattened state of the tensor contraction.
    dims = size(X)
    buffers = ntuple(Val(N-1)) do n
        len = 1
        for i in 1:n
            len *= dims[i]
        end
        Vector{T}(undef, len)
    end

    for r in 1:R
        # Contract the last mode (Mode N)
        v_N = view(factors[N], :, r)
        mat_X = reshape(X, length(buffers[N-1]), size(X, N))
        mul!(buffers[N-1], mat_X, v_N)

        # Sequentially contract modes N-1 down to 2
        for n in N-1:-1:2
            v_n = view(factors[n], :, r)
            mat_prev = reshape(buffers[n], length(buffers[n-1]), size(X, n))
            mul!(buffers[n-1], mat_prev, v_n)
        end

        # Contract mode 1
        v_1 = view(factors[1], :, r)
        total_inner += dot(buffers[1], v_1)
    end

    loss = norm2_X - 2 * total_inner + norm2_recon
    return max(loss, zero(T))
end

"""
    compute_mttkrp!(
        M::AbstractMatrix{T},
        X::AbstractArray{T, N},
        factors::NTuple{N, <:AbstractMatrix{T}},
        ::Val{n}
    ) where {T <: Number, N, n}

Computes the Matricized Tensor Times Khatri-Rao Product (MTTKRP) of an arbitrary-order
tensor `X` along mode `n`, storing the result in the preallocated matrix `M`.

This implementation avoids explicitly forming either the matricized tensor or the
Khatri-Rao product. Instead, it evaluates each rank-`r` column independently through
a sequence of tensor contractions using BLAS-backed matrix-vector multiplications.
The computation proceeds by contracting modes `N, N-1, ..., n+1` first, then
`1, 2, ..., n-1`, while keeping mode `n` uncontracted. This reduces intermediate
memory usage, enables efficient in-place execution, and is suitable for high-performance
tensor factorization workflows.

# Arguments
- `M`: Preallocated output matrix of size `(size(X, n), R)`, where `R` is the shared
  column dimension of the factor matrices.
- `X`: Input tensor of order `N`.
- `factors`: Tuple of `N` factor matrices, where `factors[k]` has size
  `(size(X, k), R)`.
- `Val(n)`: Compile-time mode index specifying the mode along which the MTTKRP is
  computed.

# Returns
- `M`: The updated output matrix containing the mode-`n` MTTKRP result, with size
  `(size(X, n), R)`.
"""
@inline function cp_loss(
    Gts::NTuple{N, <:AbstractMatrix{T}},
    A_n::AbstractMatrix{T},
    mttkrp_n::AbstractMatrix{T},
    X_norm2::T
) where {N, T <: Real}
    # ||X̂||_F^2 = sum(G₁ .* G₂ .* ... .* G_N)
    R1, R2 = size(Gts[1])
    norm2_recon = zero(T)
    Hardmard_prod = ones(T, R1, R2)
    @inbounds for i in 1:N
        G = Gts[i]
        Hardmard_prod .*= G
    end
    norm2_recon = sum(Hardmard_prod)

    loss = X_norm2 - 2 * dot(mttkrp_n, A_n) + norm2_recon
    return max(loss, zero(T))
end

"""
    compute_mttkrp!(
        M::AbstractMatrix{T},
        X::AbstractArray{T, N},
        factors::NTuple{N, <:AbstractMatrix{T}},
        ::Val{n}
    ) where {T <: Number, N, n}

Computes the Matricized Tensor Times Khatri-Rao Product (MTTKRP) of an arbitrary-order
tensor `X` along mode `n`, storing the result in the preallocated matrix `M`.

This implementation avoids explicitly forming either the matricized tensor or the
Khatri-Rao product. Instead, it evaluates each rank-`r` column independently through
a sequence of tensor contractions using BLAS-backed matrix-vector multiplications.
The computation proceeds by contracting modes `N, N-1, ..., n+1` first, then
`1, 2, ..., n-1`, while keeping mode `n` uncontracted. This reduces intermediate
memory usage, enables efficient in-place execution, and is suitable for high-performance
CPU and GPU-compatible tensor factorization workflows.

# Arguments
- `M`: Preallocated output matrix of size `(size(X, n), R)`, where `R` is the column
  dimension of the factor matrices.
- `X`: Input tensor of order `N`.
- `factors`: Tuple of `N` factor matrices, where `factors[k]` has size
  `(size(X, k), R)`.
- `Val(n)`: Compile-time mode index specifying the MTTKRP mode.

# Returns
- `M`: The updated output matrix containing the mode-`n` MTTKRP result.
"""
function compute_mttkrp!(
    M::AbstractMatrix{T},
    X::AbstractArray{T, N},
    factors::NTuple{N, <:AbstractMatrix{T}},
    ::Val{n}
) where {T <: Number, N, n}
    dims = size(X)
    R = size(factors[1], 2)
    I_n = dims[n]

    @assert size(M, 1) == I_n
    @assert size(M, 2) == R

    # Check factor sizes
    @inbounds for k in 1:N
        @assert size(factors[k], 1) == dims[k]
        @assert size(factors[k], 2) == R
    end

    # Maximum intermediate vector length needed during contractions
    max_tmp = I_n
    # After contracting modes N, N-1, ..., k we may have length prod(dims[1:k-1])
    p = 1
    for k in 1:N-1
        p *= dims[k]
        if k >= n
            max_tmp = max(max_tmp, p)
        end
    end
    # During contractions of modes 1,2,...,k-1 we may have length prod(dims[k:N])
    p = 1
    for k in N:-1:2
        p *= dims[k]
        if k <= n
            max_tmp = max(max_tmp, p)
        end
    end

    buf1 = Vector{T}(undef, max_tmp)
    buf2 = Vector{T}(undef, max_tmp)

    @views for r in 1:R
        src_is_X = true
        src = buf1
        dst = buf2
        current_len = length(X)

        # Contract modes N, N-1, ..., n+1
        # Remaining object conceptually has dims (I1, I2, ..., In)
        for k in N:-1:(n+1)
            v = view(factors[k], :, r)
            rows = current_len ÷ dims[k]

            if src_is_X
                mat = reshape(X, rows, dims[k])
                mul!(view(dst, 1:rows), mat, v)
                src_is_X = false
            else
                mat = reshape(view(src, 1:current_len), rows, dims[k])
                mul!(view(dst, 1:rows), mat, v)
            end

            current_len = rows
            src, dst = dst, src
        end

        # Contract modes 1, 2, ..., n-1
        # Keep mode n alive, so final result is length I_n
        if n == 1
            # Nothing on the left to contract; result already length I_1
            if src_is_X
                # N == 1
                copyto!(view(M, :, r), vec(X))
            else
                copyto!(view(M, :, r), view(src, 1:I_n))
            end
        else
            for k in 1:(n-1)
                v = view(factors[k], :, r)
                cols = current_len ÷ dims[k]

                if src_is_X
                    mat = reshape(X, dims[k], cols)
                    if k == n - 1
                        mul!(view(M, :, r), transpose(mat), v)
                    else
                        mul!(view(dst, 1:cols), transpose(mat), v)
                        src_is_X = false
                        current_len = cols
                        src, dst = dst, src
                    end
                else
                    mat = reshape(view(src, 1:current_len), dims[k], cols)
                    if k == n - 1
                        mul!(view(M, :, r), transpose(mat), v)
                    else
                        mul!(view(dst, 1:cols), transpose(mat), v)
                        current_len = cols
                        src, dst = dst, src
                    end
                end
            end
        end
    end

    return M
end

"""
    cp_als(
        X::AbstractArray{T, N},
        cp_rank::Int;
        max_iter::Int=10000,
        dloss_rtol::Float64=1e-6,
        loss_rtol::Float64=1e-8,
        show_trace::Bool=false,
        show_every::Int=100,
    ) where {T <: Real, N}

Computes a rank-`cp_rank` CANDECOMP/PARAFAC (CP) decomposition of an arbitrary-order
tensor `X` using alternating least squares (ALS).

This method iteratively updates each factor matrix by solving the normal equations
associated with one mode while keeping all other factors fixed. MTTKRP terms are
computed without explicitly forming matricized tensors or Khatri-Rao products, and
the Gram matrices of the factors are reused across iterations to reduce computational
cost. The relative reconstruction loss is monitored throughout the optimization, and
the iteration stops when either the loss becomes sufficiently small or the change in
loss falls below the specified tolerance.

# Arguments
- `X`: Input tensor of order `N`, with `N >= 2`.
- `cp_rank`: Target CP rank.

# Keyword Arguments
- `max_iter`: Maximum number of ALS iterations.
- `dloss_rtol`: Relative tolerance on the change in loss between successive iterations.
  Iteration stops when `abs(last_loss - loss) < dloss_rtol`.
- `loss_rtol`: Relative tolerance on the loss itself. Iteration stops when
  `loss < loss_rtol`.
- `show_trace`: If `true`, prints iteration progress and current loss.
- `show_every`: Frequency, in iterations, at which progress information is printed
  when `show_trace=true`.

# Returns
- `factors`: A tuple of `N` factor matrices, where `factors[n]` has size
  `(size(X, n), cp_rank)`.
"""
function cp_als(
    X::AbstractArray{T, N},
    cp_rank::Int;
    max_iter::Int=10000,
    dloss_rtol::Float64=1e-6,
    loss_rtol::Float64=1e-8,
    show_trace::Bool=false,
    show_every::Int=100,
) where {T<:Real, N}
    @assert N >= 2 "CPD requires at least a 2nd-order tensor."
    dims = size(X)

    # Initialize factor matrices as a Tuple of N Matrices
    factors = ntuple(n -> randn(T, dims[n], cp_rank), N)
    
    # Precompute all Gram matrices
    Gts = ntuple(n -> factors[n]' * factors[n], N)

    # Preallocate buffers for computation
    V = Matrix{T}(undef, cp_rank, cp_rank)
    mttkrp_buf = ntuple(n -> Matrix{T}(undef, dims[n], cp_rank), N)

    norm_tensor = norm(X)
    norm2_tensor = norm_tensor^2
    last_loss = sqrt(cp_loss(factors, X)) / norm_tensor

    if show_trace
        println("Iteration 0: Time = 0.0 s, Loss = $last_loss")
    end
    start_time = time()

    for iter in 1:max_iter
        for n in 1:N
            # 1. Calculate the Hadamard product V of all Gram matrices except for mode n
            first_idx = n == 1 ? 2 : 1
            copyto!(V, Gts[first_idx])
            for d in (first_idx+1):N
                d == n && continue
                V .*= Gts[d]
            end

            # Update the n-th factor matrix
            # Val(n) is used to force compile-time specialization of n, 
            # working in tandem with the generated function
            compute_mttkrp!(mttkrp_buf[n], X, factors, Val(n))
            
            copyto!(factors[n], mttkrp_buf[n])
            rdiv!(factors[n], cholesky!(Symmetric(V)))
            
            # 3. Update the corresponding Gram matrix
            mul!(Gts[n], factors[n]', factors[n])
        end

        # Quickly evaluate the current loss using the last updated dimension N
        loss = sqrt(cp_loss(Gts, factors[N], mttkrp_buf[N], norm2_tensor)) / norm_tensor

        if show_trace && iter % show_every == 0
            println("Iteration $iter: Time = $(time() - start_time) s, Loss = $loss")
        end

        stop_criterion = (abs(last_loss - loss) < dloss_rtol || loss < loss_rtol)
        if iter > 1 && stop_criterion
            show_trace && println("Converged at iteration $iter, Loss = $loss")
            break
        end
        last_loss = loss
    end

    # Return a Tuple containing all factors instead of just A, B, C
    return factors
end