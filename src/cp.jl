using LinearAlgebra
using Tullio, LoopVectorization

export cp_loss, cp_als

using LinearAlgebra

"""
    flat_to_cp_factors(
        p::AbstractVector{T},
        cp_rank::Int,
        row_sizes::NTuple{N, Int},
    ) where {T <: Real, N}

Reshapes a flat parameter vector `p` into a tuple of CP factor matrices.

This function interprets `p` as the concatenation of `N` factor matrices stored in
column-major order, where the `n`-th factor has size `(row_sizes[n], cp_rank)`.
It returns a tuple whose entries are views into the original vector reshaped as
matrices, so no data is copied during the transformation.

This is useful when CP factors are stored or optimized in flattened form, for example
in gradient-based optimization routines or parameter packing/unpacking utilities.

# Arguments
- `p`: Flat parameter vector containing all factor entries.
- `cp_rank`: Common column dimension of each factor matrix, i.e. the CP rank.
- `row_sizes`: Tuple specifying the number of rows in each factor matrix.

# Returns
- `cp_factors`: A tuple of `N` factor matrices, where `cp_factors[n]` has size
  `(row_sizes[n], cp_rank)`.
"""
@inline function flat_to_cp_factors(
    p::AbstractVector{T},
    cp_rank::Int,
    row_sizes::NTuple{N, Int}
) where {T <: Number, N}
    @assert length(p) == cp_rank * sum(row_sizes) "Length of p must match total number of entries in the factor matrices."

    idx = 1
    @inbounds cp_factors = ntuple(N) do n
        row = row_sizes[n]
        len = row * cp_rank
        factor = reshape(@view(p[idx:idx + len - 1]), row, cp_rank)
        idx += len
        factor
    end

    return cp_factors
end

"""
    cp_factors_to_flat(
        cp_factors::NTuple{N, <:AbstractMatrix{T}},
    ) where {T <: Real, N}

Flattens a tuple of CP factor matrices into a single parameter vector.

This function packs the factor matrices in `cp_factors` into one contiguous vector by
concatenating the entries of each matrix in column-major order. The factors are stored
sequentially in the same order as they appear in the input tuple, making this function
the inverse of [`flat_to_cp_factors`](@ref) when the same `cp_rank` and row sizes are
used.

This is useful when CP factors need to be represented in flattened form, for example
for gradient-based optimization, parameter serialization, or interoperability with
generic vector-based numerical routines.

# Arguments
- `cp_factors`: Tuple of `N` factor matrices, where all factors have the same number
  of columns equal to the CP rank.

# Returns
- `p`: A flat vector containing all factor entries in column-major order.
"""
@inline function cp_factors_to_flat(
    cp_factors::NTuple{N, <:AbstractMatrix{T}},
) where {T <: Real, N}
    cp_rank = size(cp_factors[1], 2)
    @assert all(size(f, 2) == cp_rank for f in cp_factors) "All factor matrices must have the same number of columns."

    total_len = sum(length, cp_factors)
    p = Vector{T}(undef, total_len)

    idx = 1
    @inbounds for factor in cp_factors
        len = length(factor)
        copyto!(p, idx, vec(factor), 1, len)
        idx += len
    end

    return p
end

"""
    cp_loss(factors::NTuple{N, AbstractMatrix{T}}, X::AbstractArray{T, N}) where {N, T <: Number}

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
    norm2_X = sum(abs2, X)

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
    cp_loss(
        Gts::NTuple{N, <:AbstractMatrix{T}},
        A_n::AbstractMatrix{T},
        mttkrp_n::AbstractMatrix{T},
        X_norm2::T,
    ) where {N, T <: Number}

Computes the squared CP reconstruction loss from precomputed Gram matrices and an
MTTKRP term for one mode.

This method is intended for use inside ALS iterations, where the Gram matrices of the
factor matrices and the mode-`n` MTTKRP term have already been computed. Rather than
re-evaluating the full loss from the factors and tensor directly, it uses the identity

`||X - X̂||_F^2 = ||X||_F^2 - 2⟨A_n, MTTKRP_n⟩ + ||X̂||_F^2`

where `||X̂||_F^2` is obtained from the Hadamard product of the Gram matrices. This
provides a fast way to monitor convergence with minimal additional cost.

# Arguments
- `Gts`: Tuple of `N` Gram matrices, where `Gts[k] = factors[k]' * factors[k]`.
- `A_n`: Factor matrix for the mode used in the MTTKRP evaluation.
- `mttkrp_n`: MTTKRP term corresponding to the same mode as `A_n`.
- `X_norm2`: Squared Frobenius norm of the input tensor, i.e. `sum(abs2, X)`.

# Returns
- The squared Frobenius loss `||X - X̂||_F^2`.
"""
@inline function cp_loss(
    Gts::NTuple{N, <:AbstractMatrix{T}},
    A_n::AbstractMatrix{T},
    mttkrp_n::AbstractMatrix{T},
    X_norm2::T
) where {N, T <: Number}
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
    cp_loss(
        A::AbstractMatrix{T},
        B::AbstractMatrix{T},
        C::AbstractMatrix{T},
        X::AbstractArray{T, 3},
    ) where {T <: Number}

Computes the CP decomposition loss for a 3rd-order tensor `X` from factor matrices
`A`, `B`, and `C` by explicitly forming the reconstruction.

This method constructs the reconstructed tensor

`X̂[i, j, k] = Σ_r A[i, r] B[j, r] C[k, r]`

and then evaluates the squared Frobenius loss `||X - X̂||_F^2`. It is primarily useful
as a straightforward reference implementation for 3-way tensors, and may also be
convenient when the explicit reconstruction is acceptable in terms of memory and
computational cost.

# Arguments
- `A`: First factor matrix of size `(size(X, 1), R)`.
- `B`: Second factor matrix of size `(size(X, 2), R)`.
- `C`: Third factor matrix of size `(size(X, 3), R)`.
- `X`: Input 3rd-order tensor.

# Returns
- The squared Frobenius loss `||X - X̂||_F^2`.
"""
@inline function cp_loss(
    A::AbstractMatrix{T},
    B::AbstractMatrix{T},
    C::AbstractMatrix{T},
    X::AbstractArray{U, 3}
) where {T <: Number, U <: Number}
    @tullio recon[i, j, k] := A[i, r] * B[j, r] * C[k, r]
    @tullio loss := (X[i, j, k] - recon[i, j, k])^2
    return loss
end

"""
    cp_loss(
        Gt_A::AbstractMatrix{T},
        Gt_B::AbstractMatrix{T},
        Gt_C::AbstractMatrix{T},
        C::AbstractMatrix{T},
        mttkrp_C::AbstractMatrix{T},
        X_norm2::T,
    ) where {T <: Number}

Computes the squared CP reconstruction loss for a 3rd-order tensor from precomputed
Gram matrices and the MTTKRP term of the third mode.

This method is a specialized 3-way variant of the Gram/MTTKRP-based loss evaluation
used inside CP-ALS. It avoids explicit tensor reconstruction by using the identity

`||X - X̂||_F^2 = ||X||_F^2 - 2⟨C, MTTKRP_C⟩ + ||X̂||_F^2`

where the reconstruction norm `||X̂||_F^2` is computed as
`sum(Gt_A .* Gt_B .* Gt_C)`. This makes the loss evaluation inexpensive during ALS
iterations.

# Arguments
- `Gt_A`: Gram matrix `A' * A`.
- `Gt_B`: Gram matrix `B' * B`.
- `Gt_C`: Gram matrix `C' * C`.
- `C`: Third-mode factor matrix.
- `mttkrp_C`: MTTKRP term corresponding to the third mode.
- `X_norm2`: Squared Frobenius norm of the input tensor, i.e. `sum(abs2, X)`.

# Returns
- The squared Frobenius loss `||X - X̂||_F^2`.
"""
@inline function cp_loss(
    Gt_A::AbstractMatrix{T},
    Gt_B::AbstractMatrix{T},
    Gt_C::AbstractMatrix{T},
    C::AbstractMatrix{T},
    mttkrp_C::AbstractMatrix{T},
    X_norm2::T
) where {T <: Number}
    loss = X_norm2 - 2 * dot(mttkrp_C, C) + sum(Gt_A .* Gt_B .* Gt_C)
    return max(loss, zero(T))
end

"""
    cp_loss(
        p::AbstractVector{T},
        cp_rank::Int,
        X::AbstractArray{U, 3},
    )::T where {T <: Real, U <: Real}

Computes the CP decomposition loss for a 3rd-order tensor `X` from a flattened
parameter vector `p`.

This function first interprets `p` as the concatenation of three CP factor matrices
with column dimension `cp_rank`, reshaping it into factors `A`, `B`, and `C` using
[`flat_to_cp_factors`](@ref). It then evaluates the squared Frobenius reconstruction
loss `||X - X̂||_F^2` by calling the corresponding 3-way [`cp_loss`](@ref) method.

This representation is convenient when CP factors are optimized in flattened form,
for example in first-order or second-order vector-based optimization routines.

# Arguments
- `p`: Flat parameter vector encoding the factor matrices `A`, `B`, and `C` in
  column-major order.
- `cp_rank`: CP rank, i.e. the common number of columns of the factor matrices.
- `X`: Input 3rd-order tensor.

# Returns
- The squared Frobenius loss `||X - X̂||_F^2`, where `X̂` is the CP reconstruction
  induced by the factors stored in `p`.
"""
function cp_loss(
    p::AbstractVector{T},
    cp_rank::Int,
    X::AbstractArray{U, 3}
)::T where {T <: Real, U <: Real}
    A, B, C = flat_to_cp_factors(p, cp_rank, size(X))
    loss = cp_loss(A, B, C, X)
    return loss
end

"""
    cp_loss_grad!(
        g::AbstractVector{T},
        p::AbstractVector{T},
        cp_rank::Int,
        tensor::AbstractArray{U, 3},
    ) where {T <: Real, U <: Real}

Computes the gradient of the 3rd-order CP reconstruction loss with respect to a
flattened parameter vector `p`, and writes the result in-place to `g`.

This function interprets `p` as the flattened CP factors `A`, `B`, and `C`, and
interprets `g` as storage for the corresponding factor gradients `gA`, `gB`, and
`gC`. The gradient is evaluated from the analytic derivative of the squared
Frobenius loss `||X - X̂||_F^2`, using MTTKRP terms and Gram matrices of the factor
matrices. No new flattened gradient vector is allocated; instead, the provided
buffer `g` is overwritten in-place.

This is useful in optimization workflows where CP factors are represented as a
single parameter vector and gradients must be supplied in the same flattened format.

# Arguments
- `g`: Output gradient vector, overwritten in-place with the gradient of the loss
  with respect to `p`.
- `p`: Flat parameter vector encoding the factor matrices in column-major order.
- `cp_rank`: CP rank, i.e. the common number of columns of the factor matrices.
- `tensor`: Input 3rd-order tensor.

# Returns
- `nothing`.
"""
function cp_loss_grad!(
    g::AbstractVector{T},
    p::AbstractVector{T},
    cp_rank::Int,
    tensor::AbstractArray{U,3}
) where {T<: Real, U <: Real}
    A, B, C = flat_to_cp_factors(p, cp_rank, size(tensor))
    gA, gB, gC = flat_to_cp_factors(g, cp_rank, size(tensor))

    GtA = A' * A
    GtB = B' * B
    GtC = C' * C

    @tullio mttkrp_A[i, r] := tensor[i, j, k] * B[j, r] * C[k, r]
    @tullio mttkrp_B[j, r] := tensor[i, j, k] * A[i, r] * C[k, r]
    @tullio mttkrp_C[k, r] := tensor[i, j, k] * A[i, r] * B[j, r]

    gA .= -2 .* mttkrp_A .+ 2 .* (A * (GtB .* GtC))
    gB .= -2 .* mttkrp_B .+ 2 .* (B * (GtA .* GtC))
    gC .= -2 .* mttkrp_C .+ 2 .* (C * (GtA .* GtB))

    return nothing
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
function mttkrp!(
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

    n_threads = Threads.maxthreadid()
    thread_buf1 = [Vector{T}(undef, max_tmp) for _ in 1:n_threads]
    thread_buf2 = [Vector{T}(undef, max_tmp) for _ in 1:n_threads]

    BLAS_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1) # Avoid oversubscription with multi-threaded BLAS
    @inbounds @views @Threads.threads for r in 1:R
        src_is_X = true
        tid = Threads.threadid()
        src = thread_buf1[tid]
        dst = thread_buf2[tid]
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
    BLAS.set_num_threads(BLAS_threads) # Restore original BLAS thread count

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
    dloss_rtol::Float64=1e-8,
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
            # Calculate the Hadamard product V of all Gram matrices except for mode n
            first_idx = n == 1 ? 2 : 1
            copyto!(V, Gts[first_idx])
            for d in (first_idx+1):N
                d == n && continue
                V .*= Gts[d]
            end

            # Update the n-th factor matrix
            # Val(n) is used to force compile-time specialization of n, 
            # working in tandem with the generated function
            mttkrp!(mttkrp_buf[n], X, factors, Val(n))
            
            copyto!(factors[n], mttkrp_buf[n])
            rdiv!(factors[n], cholesky!(Symmetric(V)))
            
            # Update the corresponding Gram matrix
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

"""
    cp_als(
        tensor::AbstractArray{T, 3},
        cpd_rank::Int;
        max_iter::Int=10000,
        dloss_rtol::Float64=1e-7,
        loss_rtol::Float64=1e-8,
        show_trace::Bool=false,
        show_every::Int=100,
    ) where {T <: Real}

Computes a rank-`cpd_rank` CANDECOMP/PARAFAC (CP) decomposition of a 3rd-order
tensor `tensor` using alternating least squares (ALS).

This method is a specialized implementation for 3-way tensors. It iteratively updates
the factor matrices `A`, `B`, and `C` by solving the normal equations for each mode
while holding the other two factors fixed. The MTTKRP terms are formed directly using
tensor contractions, and the Gram matrices of the factors are reused across iterations
to avoid redundant computation. The relative reconstruction loss is monitored during
optimization, and the iteration terminates when either the loss becomes sufficiently
small or the change in loss between successive iterations falls below the specified
tolerance.

# Arguments
- `tensor`: Input 3rd-order tensor of size `(I, J, K)`.
- `cpd_rank`: Target CP rank.

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
- `A`: Factor matrix of size `(size(tensor, 1), cpd_rank)`.
- `B`: Factor matrix of size `(size(tensor, 2), cpd_rank)`.
- `C`: Factor matrix of size `(size(tensor, 3), cpd_rank)`.
"""
function cp_als(
    tensor::AbstractArray{T,3},
    cpd_rank::Int;
    max_iter::Int=10000,
    dloss_rtol::Float64=1e-8,
    loss_rtol::Float64=1e-8,
    show_trace::Bool=false,
    show_every::Int=100,
) where {T<:Real}
    I, J, K = size(tensor)

    A = randn(T, I, cpd_rank)
    B = randn(T, J, cpd_rank)
    C = randn(T, K, cpd_rank)
    mttkrp_C = Matrix{T}(undef, K, cpd_rank)
    V = Matrix{T}(undef, cpd_rank, cpd_rank)    # For Hardamard product of Gram matrices

    GtA = Matrix{T}(undef, cpd_rank, cpd_rank)
    GtB = Matrix{T}(undef, cpd_rank, cpd_rank)
    GtC = Matrix{T}(undef, cpd_rank, cpd_rank)

    mul!(GtB, B', B)
    mul!(GtC, C', C)
    
    norm_tensor = norm(tensor)
    norm2_tensor = norm_tensor^2
    last_loss = sqrt(cp_loss(A, B, C, tensor)) / norm_tensor

    if show_trace
        println("Iteration 0: Time = 0.0 s, Loss = $last_loss")
    end
    start_time = time()
    for iter in 1:max_iter
        # Update A
        @tullio A[i, r] = tensor[i, j, k] * B[j, r] * C[k, r]
        @. V = GtB * GtC
        rdiv!(A, cholesky!(Symmetric(V)))  # Solve A * V_A = mttkrp_A
        mul!(GtA, A', A)

        # Update B
        @tullio B[j, r] = tensor[i, j, k] * A[i, r] * C[k, r]
        @. V = GtA * GtC
        rdiv!(B, cholesky!(Symmetric(V)))
        mul!(GtB, B', B)

        # Update C
        @tullio mttkrp_C[k, r] = tensor[i, j, k] * A[i, r] * B[j, r]
        @. V = GtA * GtB
        copyto!(C, mttkrp_C)
        rdiv!(C, cholesky!(Symmetric(V)))
        mul!(GtC, C', C)

        # Evaluate loss
        loss = sqrt(cp_loss(GtA, GtB, GtC, C, mttkrp_C, norm2_tensor)) / norm_tensor

        if show_trace && iter % show_every == 0
            println("Iteration $iter: Time = $(time() - start_time) s, Loss = $loss")
        end

        stop_criterion = (abs(last_loss - loss) < dloss_rtol
                          || loss < loss_rtol)
        if iter > 1 && stop_criterion
            show_trace && println("Converged at iteration $iter, Loss = $loss")
            break
        end
        last_loss = loss
    end

    return A, B, C
end