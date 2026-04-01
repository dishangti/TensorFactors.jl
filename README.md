# TensorFactors.jl
An alternative tensor factorization toolbox designed for high performance.

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://dishangti.github.io/TensorFactors.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://dishangti.github.io/TensorFactors.jl/dev/)
[![Build Status](https://github.com/dishangti/TensorFactors.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/dishangti/TensorFactors.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/dishangti/TensorFactors.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/dishangti/TensorFactors.jl)

## CP Tensor Decomposition

This package provides high-performance, memory-efficient routines for computing the **CANDECOMP/PARAFAC (CP)** decomposition of multi-dimensional arrays (tensors) in Julia. 

The CP decomposition factorizes a tensor into a sum of component rank-one tensors. For a 3rd-order tensor $X$, this approximates the tensor as:
$$X \approx \sum_{r=1}^{R} \lambda_r (a_r \circ b_r \circ c_r)$$
where $R$ is the CP rank, $\lambda$ contains the normalizing weights, and $A, B, C$ are the factor matrices.

This implementation leverages pure BLAS operations and fast tensor contractions via `Tullio.jl` and `LoopVectorization.jl` to evaluate Matricized Tensor Times Khatri-Rao Products (MTTKRP) and reconstruction losses *without* explicitly allocating intermediate matricized tensors. 

### Features

* **`cp_als`**: Alternating Least Squares (ALS) optimization. Features generalized N-way tensor support and highly optimized, specialized routines for 3-way tensors.
* **`cp_opt`**: All-at-once gradient-based optimization leveraging `Optim.jl`. Optimizes flattened parameter vectors using exact analytic gradients.
* **Low Memory Footprint**: Bypasses explicit Khatri-Rao product formation and full tensor reconstruction during loss evaluation.

### Usage Examples

#### 1. Alternating Least Squares (`cp_als`)

The ALS algorithm is the workhorse for CP decomposition. It iteratively optimizes one factor matrix at a time while keeping the others fixed. It is generally the fastest way to get a good decomposition.

```julia
using LinearAlgebra
using Tullio, LoopVectorization

# 1. Create a synthetic 3rd-order tensor (e.g., 20 x 30 x 50 with CPD rank 10)
I, J, K, cp_rank = 20, 30, 50, 10
A, B, C = randn(I, R), randn(J, R), randn(K, R)
@tullio X[i, j, k] := A[i, r] * B[j, r] * C[k, r]

# 3. Run the CP-ALS algorithm
# The function monitors the relative reconstruction loss and stops 
# when the change falls below `dloss_rtol` (default 1e-8).
λ, A, B, C = cp_als(
    X, 
    cp_rank; 
    max_iter=500, 
    show_trace=true, 
    show_every=10
)

println("Decomposition complete!")
println("Weights (λ): ", size(λ))
println("Factor A: ", size(A))
println("Factor B: ", size(B))
println("Factor C: ", size(C))
```
*Note: Weights can be obsorbed in to one of the factor as `A .* λ'`. For tensors of order $N > 3$, `cp_als` returns `(λ, factors)` where `factors` is an N-tuple of factor matrices.*


#### 2. Gradient-Based Optimization (`cp_opt`)

For scenarios where ALS struggles (e.g., "swamps" or highly collinear factors), direct optimization of the loss function using first-order or quasi-Newton methods can be highly effective. The `cp_opt` function wraps solvers from `Optim.jl`, and uses analytic gradient for efficiency.

```julia
using Optim

# 1. Choose an optimizer from Optim.jl (ConjugateGradient is highly recommended for this)
optimizer = ConjugateGradient()

# 2. Run the optimization
# You can optionally pass `init_factors` if you want to warm-start 
# the optimization (e.g., using the output from a few iterations of cp_als)
factors_opt = cp_opt(
    optimizer, 
    X, 
    cp_rank; 
    max_iter=200, 
    show_trace=true, 
    show_every=10
)

# Unpack the resulting factor matrices
A_opt, B_opt, C_opt = factors_opt
```

## Tucker Tensor Decomposition

This package provides highly optimized routines for computing the **Tucker Decomposition** (often associated with Higher-Order Principal Component Analysis) of $N$-dimensional arrays in Julia. 

The Tucker decomposition factorizes a tensor into a dense, compressed "core" tensor multiplied by an orthogonal factor matrix along each mode. For an $N$-order tensor $\mathcal{X}$, this approximates the tensor as:
$$\mathcal{X} \approx \mathcal{S} \times_1 U^{(1)} \times_2 U^{(2)} \dots \times_N U^{(N)}$$
where $\mathcal{S}$ is the core tensor, $U^{(n)}$ are the orthogonal factor matrices, and $\times_n$ denotes the $n$-mode product (Tensor-Times-Matrix).

This implementation is heavily tuned for performance. It completely bypasses the computational bottlenecks of standard SVDs on massively wide unfolded matrices by leveraging the symmetric Gram matrix trick ($A_{(n)}A_{(n)}^T$) combined with pure BLAS `SYRK` routines. Memory allocations are strictly minimized through intelligent buffer pooling and type-stable `Tuple` permutations.

### Features

* **`tucker_hosvd`**: Computes the Truncated Higher-Order Singular Value Decomposition (HOSVD). Provides lightning-fast, deterministic compression of high-dimensional data to specified target ranks.
* **`ttm`**: $n$-mode product (Tensor-Times-Matrix) operations. Enforces contiguous memory layouts to guarantee maximum BLAS matrix multiplication throughput.

### Usage Examples

#### 1. Truncated HOSVD (`hosvd`)

The HOSVD is the standard method for computing a Tucker decomposition. It is highly effective for data compression, dimensionality reduction, and feature extraction in multi-way data.

```julia
using LinearAlgebra
using TensorFactors # Replace with your actual package name

# Create a synthetic 3rd-order tensor (e.g., 100 x 150 x 200)
I, J, K = 100, 150, 200
X = randn(I, J, K)

# Specify the target truncation ranks for each mode
target_ranks = (10, 15, 20)

# Run the Truncated HOSVD algorithm
# S is the compressed core tensor, U is an N-Tuple of factor matrices
S, U = tucker_hosvd(X, target_ranks)

println("Tucker Decomposition complete!")
println("Core Tensor (S): ", size(S)) # Output: (10, 15, 20)
println("Factor U1: ", size(U[1]))    # Output: (100, 10)
println("Factor U2: ", size(U[2]))    # Output: (150, 15)
println("Factor U3: ", size(U[3]))    # Output: (200, 20)
```