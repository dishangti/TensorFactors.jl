using LinearAlgebra

export tucker_hosvd, ttm, tucker_contract

"""
    unfold_mode(
        A::AbstractArray{T, N},
        mode::Int
    ) where {T, N}

Unfolds (matricizes) an `N`-dimensional tensor `A` into a matrix along the specified `mode`.

This function permutes the dimensions of the tensor so that the target `mode` becomes 
the first dimension, while preserving the relative order of the remaining dimensions. 
It then reshapes the result into a 2D matrix. The returned matrix is explicitly 
materialized into a contiguous array (rather than a lazy view) to guarantee optimal 
BLAS performance in downstream matrix multiplications.

# Arguments
- `A`: The `N`-dimensional input tensor to be unfolded.
- `mode`: The integer specifying the dimension along which to unfold the tensor.

# Returns
- `unfolded_matrix`: A 2D matrix of size `(size(A, mode), J)`, where `J` is the product 
  of all other dimensions of the tensor.
"""
function unfold_mode(
    A::AbstractArray{T, N},
    mode::Int
) where {T, N}
    sz = size(A)
    p = [mode; setdiff(1:N, mode)]  # Permutation to bring the target mode to the front
    A_perm = permutedims(A, p)
    return reshape(A_perm, sz[mode], :)
end

"""
    ttm(
        A::AbstractArray{T, N},
        M::AbstractMatrix,
        mode::Int
    ) where {T, N}

Computes the `n`-mode product (Tensor-Times-Matrix) of a tensor `A` with a matrix `M`.

This operation projects the `mode`-th dimension of the tensor `A` onto the row space 
of the matrix `M`. Internally, the function unfolds the tensor along `mode`, performs 
a standard matrix multiplication `M * A_{(n)}`, reshapes the result, and restores the 
original order of the dimensions. 

# Arguments
- `A`: The `N`-dimensional input tensor.
- `M`: The matrix to multiply with the tensor. The number of columns in `M` must 
  exactly match the length of `A`'s `mode`-th dimension (`size(M, 2) == size(A, mode)`).
- `mode`: The integer specifying the dimension of the tensor to be multiplied.

# Returns
- `res`: A new `N`-dimensional tensor. Its size is identical to `A`, except for the 
  `mode`-th dimension, which is updated to the number of rows in `M` (`size(M, 1)`).
"""
function ttm(
    A::AbstractArray{T, N},
    M::AbstractMatrix,
    mode::Int
) where {T, N}
    sz = size(A)
    p = Tuple([mode; setdiff(1:N, mode)])
    
    A_unfolded = unfold_mode(A, mode)
    res_unfolded = M * A_unfolded
    
    new_sz = Tuple([size(M, 1); [sz[i] for i in p[2:end]]])
    res_perm = reshape(res_unfolded, new_sz...)
    
    inv_p = invperm(p)  # Permutation to restore original dimension order
    return permutedims(res_perm, inv_p)
end

"""
    hosvd(
        A::AbstractArray{T,N},
        ranks::NTuple{N,Int}
    ) where {T <: Number,N}

Computes the Truncated Higher-Order Singular Value Decomposition (HOSVD) of tensor `A`.

This function decomposes an `N`-dimensional tensor into a compressed core tensor and 
a set of orthogonal factor matrices corresponding to each mode. To maximize performance 
and minimize memory overhead, it avoids computing full SVDs on highly wide unfolded 
matrices. Instead, it computes the symmetric Gram matrix `A_{(n)} * A_{(n)}'` utilizing 
fast BLAS `SYRK` routines, followed by an eigenvalue decomposition. Memory allocations 
are further optimized by pooling matrix buffers based on the tensor's unique dimensions.

# Arguments
- `A`: The `N`-dimensional input tensor to be decomposed.
- `ranks`: An `N`-element tuple of integers specifying the target truncation rank 
  for each respective mode.

# Returns
- `S`: The highly compressed `N`-dimensional core tensor of size `ranks`.
- `factors`: A tuple of `N` orthogonal factor matrices. The `n`-th matrix, 
  `factors[n]`, has a size of `(size(A, n), ranks[n])`.
"""
function tucker_hosvd(
    A::AbstractArray{T,N},
    ranks::NTuple{N,Int}
) where {T <: Number,N}
    # Preallocate Gram matrix buffers for each unique dimension size to minimize allocations
    sz = size(A)
    sz_set = sort(unique(sz))
    sz_idx = indexin(sz, sz_set)
    gram_buffers = [similar(A, sz_set[i], sz_set[i]) for i in eachindex(sz_set)]
    factors = ntuple(N) do n
        gram_buffer = gram_buffers[sz_idx[n]]
        An = unfold_mode(A, n)
        r = ranks[n]

        mul!(gram_buffer, An, An')
        F = eigen(Symmetric(gram_buffer))
        
        # Take the top-r eigenvectors corresponding to the largest eigenvalues
        U_n = F.vectors[:, end:-1:end-r+1]
        
        return U_n
    end
    
    # Compute the core tensor
    S = A
    for n in 1:N
        S = ttm(S, factors[n]', n)
    end
    
    return S, factors
end

"""
    tucker_contract(core::AbstractArray{T, N}, factors::NTuple{N, <:AbstractMatrix{T}}) where {T <: Number, N}

Contract a Tucker decomposition, consisting of a core tensor and `N` factor matrices, into a full tensor.

This function reconstructs a full `N`-dimensional tensor by multi-linear product of a 
core tensor `G` with `N` factor matrices. To maximize performance and ensure correct 
memory allocation, the implementation utilizes metaprogramming to generate specialized 
methods for `N` ranging from 3 to 10. Each method leverages `@tullio` for efficient, 
multi-threaded tensor contraction. By using metaprogramming to unpack the factor 
matrices into distinct variables, the `@tullio` symbolic analyzer can generate 
highly optimized kernel loops without the overhead of indexing into a collection.

# Arguments
- `core`: An `N`-dimensional core tensor of size (r1, r2, ..., rN).
- `factors`: An `N`-element tuple of matrices, where each matrix `factors[n]` 
  of size (In, rn) represents the factor matrix for the `n`-th mode.

# Returns
- `X`: The reconstructed `N`-dimensional tensor of size (I1, I2, ..., IN).
"""
function tucker_contract end

for N in 3:10
    # Generate variable names for factor matrices: [:A1, :A2, ..., :AN]
    matrix_vars = [Symbol("A", d) for d in 1:N]
    
    # Generate index symbols for the output tensor: [:i1, :i2, ..., :iN]
    out_idx = [Symbol("i", d) for d in 1:N]
    
    # Generate index symbols for the core tensor (summation indices): [:r1, :r2, ..., :rN]
    core_idx = [Symbol("r", d) for d in 1:N]

    # Build the factor matrix multiplication terms: [A1[i1, r1], A2[i2, r2], ...]
    matrix_terms = [:( $(matrix_vars[d])[$(out_idx[d]), $(core_idx[d])] ) for d in 1:N]
    
    # Build the core tensor term: G[r1, r2, ..., rN]
    core_term = :( G[$(core_idx...)] )
    
    # Combine all terms into the product for the Right-Hand Side (RHS)
    # Result: G[r1, ..., rN] * A1[i1, r1] * A2[i2, r2] * ...
    rhs = Expr(:call, :*, core_term, matrix_terms...)

    # Build the left-hand side tuple unpacking for factor matrices: (A1, A2, ..., AN)
    unpack_factors = Expr(:tuple, matrix_vars...)

    @eval begin
        function tucker_contract(core::AbstractArray{T, $N}, factors::NTuple{$N, <:AbstractMatrix{T}}) where {T <: Number}
            G = core
            $unpack_factors = factors
            @tullio X[$(out_idx...)] := $rhs
            return X
        end
    end
end