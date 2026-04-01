using LinearAlgebra
using Tullio, LoopVectorization
using Optim

export cp_loss, cp_loss_grad!, cp_als, cp_opt, cp_contract

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
    factors::NTuple{3, AbstractMatrix{T}},
    X::AbstractArray{U, 3}
) where {T <: Number, U <: Number}
    A, B, C = factors
    @tullio recon[i, j, k] := A[i, r] * B[j, r] * C[k, r]
    @tullio loss := (X[i, j, k] - recon[i, j, k])^2
    return loss
end

@inline function cp_loss(
    factors::NTuple{3, AbstractMatrix{T}},
    X::AbstractArray{U, 3},
    X_norm2::V
) where {T <: Number, U <: Number, V <: Real}
    A, B, C = factors
    @tullio mttkrp_C[k, r] := X[i, j, k] * A[i, r] * B[j, r]
    Gt_A = A' * A
    Gt_B = B' * B
    Gt_C = C' * C
    loss = X_norm2 - 2 * dot(mttkrp_C, C) + sum(Gt_A .* Gt_B .* Gt_C)
    return max(loss, zero(T))
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
        X::AbstractArray{U, 3},
        cp_rank::Int
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
- `X`: Input 3rd-order tensor.
- `cp_rank`: CP rank, i.e. the common number of columns of the factor matrices.

# Returns
- The squared Frobenius loss `||X - X̂||_F^2`, where `X̂` is the CP reconstruction
  induced by the factors stored in `p`.
"""
function cp_loss(
    p::AbstractVector{T},
    X::AbstractArray{U, 3},
    cp_rank::Int,
    norm2_X::V
)::T where {T <: Real, U <: Real, V <: Real}
    A, B, C = flat_to_cp_factors(p, cp_rank, size(X))
    loss = cp_loss((A, B, C), X, norm2_X)
    return loss
end

"""
    cp_loss_grad!(
        g::AbstractVector{T},
        p::AbstractVector{T},
        X::AbstractArray{U, 3},
        cp_rank::Int
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
- `X`: Input 3rd-order tensor.
- `cp_rank`: CP rank, i.e. the common number of columns of the factor matrices.

# Returns
- `nothing`.
"""
function cp_loss_grad!(
    g::AbstractVector{T},
    p::AbstractVector{T},
    X::AbstractArray{U,3},
    cp_rank::Int
) where {T<: Real, U <: Real}
    A, B, C = flat_to_cp_factors(p, cp_rank, size(X))
    gA, gB, gC = flat_to_cp_factors(g, cp_rank, size(X))

    GtA = A' * A
    GtB = B' * B
    GtC = C' * C

    H_A = GtB .* GtC
    H_B = GtA .* GtC
    H_C = GtA .* GtB

    @tullio gA[i, r] = X[i, j, k] * B[j, r] * C[k, r]
    @tullio gB[j, r] = X[i, j, k] * A[i, r] * C[k, r]
    @tullio gC[k, r] = X[i, j, k] * A[i, r] * B[j, r]

    gA .= -2 .* gA .+ 2 .* (A * H_A)
    gB .= -2 .* gB .+ 2 .* (B * H_B)
    gC .= -2 .* gC .+ 2 .* (C * H_C)

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
    colnormalize!(
        A::AbstractMatrix{T},
        lambda::AbstractVector{T},
    ) where {T <: Number}

Normalizes each column of matrix `A` in place and absorbs the corresponding column
norms into the weight vector `lambda`.

For each column `r`, this method computes its Euclidean norm. If the norm is positive,
the column is rescaled to have unit norm, and `lambda[r]` is multiplied by the original
column norm.

This routine is commonly used in tensor factorization algorithms to separate per-column
scaling factors from factor matrices while keeping the product represented by
`A` and `lambda` unchanged.

# Arguments
- `A`: Input matrix whose columns are normalized in place.
- `lambda`: Vector of column weights. Its length must equal `size(A, 2)`.

# Returns
- `A, lambda`: The normalized matrix `A` and the updated weight vector `lambda`.
"""
function colnormalize!(
    A::AbstractMatrix{T},
    lambda::AbstractVector{T},
) where {T<:Number}
    R = size(A, 2)
    @assert length(lambda) == R

    @inbounds for r in 1:R
        col = view(A, :, r)
        nrm = norm(col)
        rmul!(col, inv(nrm))
        lambda[r] = nrm
    end

    return A, lambda
end

"""
    colnormalize!(A::AbstractMatrix{T}) where {T <: Number}

Normalizes each column of matrix `A` in place to have unit Euclidean norm.

For each column `r`, this method computes its Euclidean norm. If the norm is positive,
the column is rescaled in place so that its 2-norm becomes `one(T)`.

This overload is useful when only column normalization is needed and no separate
weight vector is maintained.

# Arguments
- `A`: Input matrix whose columns are normalized in place.

# Returns
- `A`: The normalized matrix `A`.
"""
function colnormalize!(
    A::AbstractMatrix{T}
) where {T<:Number}
    R = size(A, 2)
    @inbounds for r in 1:R
        col = view(A, :, r)
        nrm = norm(col)
        rmul!(col, inv(nrm))
    end

    return A
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

    # Preallocate the lambda vector for column normalization
    lambda = ones(T, cp_rank)

    # Buffer for loss check
    loss_GtN = Matrix{T}(undef, cp_rank, cp_rank)
    loss_factorN = Matrix{T}(undef, dims[N], cp_rank)

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
            colnormalize!(factors[n], lambda)  # Absorb scaling into lambda to prevent numerical issues

            # Update the corresponding Gram matrix
            mul!(Gts[n], factors[n]', factors[n])
        end

        # Quickly evaluate the current loss using the last updated dimension N
        @. loss_factorN = factors[N] .* lambda'
        copyto!(loss_GtN, Gts[N])  # Backup GtN before overwriting for loss evaluation
        mul!(Gts[N], loss_factorN', loss_factorN)  # Ensure GtN is up to date for loss evaluation
        loss = sqrt(cp_loss(Gts, loss_factorN, mttkrp_buf[N], norm2_tensor)) / norm_tensor
        copyto!(Gts[N], loss_GtN)  # Restore GtN after loss evaluation

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
    return lambda, factors
end

"""
    cp_als(
        X::AbstractArray{T, 3},
        cp_rank::Int;
        max_iter::Int=10000,
        dloss_rtol::Float64=1e-7,
        loss_rtol::Float64=1e-8,
        show_trace::Bool=false,
        show_every::Int=100,
    ) where {T <: Real}

Computes a rank-`cp_rank` CANDECOMP/PARAFAC (CP) decomposition of a 3rd-order
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
- `X`: Input 3rd-order tensor of size `(I, J, K)`.
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
- `A`: Factor matrix of size `(size(X, 1), cp_rank)`.
- `B`: Factor matrix of size `(size(X, 2), cp_rank)`.
- `C`: Factor matrix of size `(size(X, 3), cp_rank)`.
"""
function cp_als(
    X::AbstractArray{T,3},
    cp_rank::Int;
    max_iter::Int=10000,
    dloss_rtol::Float64=1e-8,
    loss_rtol::Float64=1e-8,
    show_trace::Bool=false,
    show_every::Int=100,
) where {T<:Real}
    I, J, K = size(X)

    A = randn(T, I, cp_rank)
    B = randn(T, J, cp_rank)
    C = randn(T, K, cp_rank)
    mttkrp_C = Matrix{T}(undef, K, cp_rank)
    V = Matrix{T}(undef, cp_rank, cp_rank)    # For Hardamard product of Gram matrices

    GtA = Matrix{T}(undef, cp_rank, cp_rank)
    GtB = Matrix{T}(undef, cp_rank, cp_rank)
    GtC = Matrix{T}(undef, cp_rank, cp_rank)

    C_loss = Matrix{T}(undef, K, cp_rank)  # Buffer for MTTKRP of C to evaluate loss
    GtC_loss = Matrix{T}(undef, cp_rank, cp_rank)  # Buffer for GtC to evaluate loss

    lambda = ones(T, cp_rank)   # Normalization of factors to prevent numerical issues

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
        loss = sqrt(cp_loss(GtA, GtB, GtC_loss, C_loss, mttkrp_C, norm2_tensor)) / norm_tensor

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

"""
    cp_opt(
        method::Optim.AbstractOptimizer,
        X::AbstractArray{T, N},
        cp_rank::Int;
        max_iter::Int=typemax(Int),
        show_trace::Bool=false,
        show_every::Int=100,
        p0::Union{Nothing, AbstractVector{T}}=nothing,
    ) where {T <: Real, N}

Fits a rank-`cp_rank` CP decomposition to a tensor `X` by minimizing the CP
reconstruction loss with an optimizer from `Optim.jl`.

This function represents the CP factor matrices as a single flattened parameter vector
and solves the resulting unconstrained optimization problem using the optimizer
specified by `method`. If no initial parameter vector is provided, one is initialized
randomly from a standard normal distribution. After optimization, the minimizer is
reshaped into a tuple of CP factor matrices using [`flat_to_cp_factors`](@ref).

The objective minimized is the squared Frobenius reconstruction loss
`||X - X̂||_F^2`, where `X̂` is the rank-`cp_rank` CP reconstruction induced by the
optimized factor matrices.

# Arguments
- `method`: Optimizer from `Optim.jl`, such as `LBFGS()` or `ConjugateGradient()`.
- `X`: Input tensor.
- `cp_rank`: Target CP rank.

# Keyword Arguments
- `max_iter`: Maximum number of optimization iterations.
- `show_trace`: If `true`, prints optimization progress information.
- `show_every`: Frequency, in iterations, at which progress information is printed
  when `show_trace=true`.
- `p0`: Optional initial flattened parameter vector. If `nothing`, a random
  initialization of length `cp_rank * sum(size(X))` is used.

# Returns
- `cp_factors`: A tuple of factor matrices defining the fitted CP decomposition,
  where `cp_factors[n]` has size `(size(X, n), cp_rank)`.
"""
function cp_opt(
    method::Optim.AbstractOptimizer,
    X::AbstractArray{T, 3},
    cp_rank::Int;
    max_iter::Int = 10000,
    show_trace::Bool = false,
    show_every::Int = 100,
    init_factors::Union{Nothing, NTuple{3, AbstractMatrix{T}}} = nothing,
) where {T <: Real}
    if init_factors === nothing
        p0 = randn(T, cp_rank * sum(size(X)))
    else
        p0 = cp_factors_to_flat(init_factors)
    end

    norm2_X = sum(abs2, X)
    f(u) = cp_loss(u, X, cp_rank, norm2_X)
    g!(g, u) = cp_loss_grad!(g, u, X, cp_rank)
    
    od = OnceDifferentiable(f, g!, p0)
    options = Optim.Options(iterations = max_iter, show_trace = show_trace, show_every = show_every)
    sol = optimize(od, p0, method, options)
    
    minimizer = Optim.minimizer(sol)

    cp_factors = flat_to_cp_factors(minimizer, cp_rank, size(X))
    return cp_factors
end

"""
    cp_contract(factors::NTuple{N, <:AbstractMatrix{T}}) where {T <: Number, N}

Contract a CP (Canonical Polyadic) decomposition with `N` factor matrices into a full tensor.

This function reconstructs a full `N`-dimensional tensor from its factor matrices. To 
maximize performance and ensure correct memory allocation, the implementation utilizes 
metaprogramming to generate specialized methods for `N` ranging from 2 to 10. Each 
method leverages `@tullio` for efficient, multi-threaded tensor contraction, ensuring 
that intermediate allocations are optimized by providing concrete matrix variables to 
the macro's symbolic analyzer.

# Arguments
- `factors`: An `N`-element tuple of matrices, where each matrix `factors[n]` 
  represents the factor matrix for the `n`-th mode.

# Returns
- `X`: The reconstructed `N`-dimensional tensor.
"""
function cp_contract end

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
        function cp_contract(factors::NTuple{$N, <:AbstractMatrix{T}}) where {T <: Number}
            $unpack_tuple = factors
            @tullio X[$(idx...)] := $rhs
            return X
        end
    end
end