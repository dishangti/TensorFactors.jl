using TensorFactors
using LinearAlgebra
using Tullio
using CUDA, KernelAbstractions
using Optim

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
        function TensorFactors.tucker_contract(core::CuArray{T, $N}, factors::NTuple{$N, <:CuArray{T, 2}}) where {T <: Number}
            G = core
            $unpack_factors = factors
            @tullio X[$(out_idx...)] := $rhs
            return X
        end
    end
end