using LinearAlgebra

export hosvd, ttm

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
    A_perm = PermutedDimsArray(A, p)
    return Array(reshape(A_perm, sz[mode], :))
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
function hosvd(
    A::AbstractArray{T,N},
    ranks::NTuple{N,Int}
) where {T <: Number,N}
    # Preallocate Gram matrix buffers for each unique dimension size to minimize allocations
    sz = size(A)
    sz_set = sort(unique(sz))
    sz_idx = indexin(sz, sz_set)
    gram_buffers = [Matrix{T}(undef, sz_set[i], sz_set[i]) for i in eachindex(sz_set)]
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