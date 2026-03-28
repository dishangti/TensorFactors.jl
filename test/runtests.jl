using TensorFactors
using LinearAlgebra
using Tullio, LoopVectorization
using Random
using ForwardDiff
using Optim
using Test

function test_cp()
    function khatri_rao!(KR::AbstractMatrix{T}, C::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T}
        K, R = size(C)
        J, R2 = size(B)
        @inbounds for r in 1:R
            KR[:, r] = vec(B[:, r] * C[:, r]')
        end
        return KR
    end

    @testset "cp.jl" begin
        Random.seed!(42)

        # Test column normalization
        A = randn(20, 10)
        TensorFactors.colnormalize!(A)
        all_norms = vec(norm.(eachcol(A)))
        @test all(isfinite.(all_norms)) && all(isapprox.(all_norms, 1.0; rtol=1e-12))
        
        # Test column normalization with lambda obsorbed
        A = randn(20, 10)
        A_copy = copy(A)
        lambda = Vector{Float64}(undef, size(A, 2))
        TensorFactors.colnormalize!(A, lambda)
        all_norms = vec(norm.(eachcol(A)))
        A_hat = A .* lambda'
        @test (all(isfinite.(all_norms))
            && all(isapprox.(all_norms, 1.0; rtol=1e-12))
            && norm(A_hat - A_copy) / norm(A_copy) < 1e-12)

        # Test cp_als on the synthetic tensor with 4-tensor
        I, J, K, L, R = 20, 30, 50, 60, 10
        A, B, C, D = randn(I, R), randn(J, R), randn(K, R), randn(L, R)
        X = zeros(I, J, K, L)
        @tullio X[i, j, k, l] := A[i, r] * B[j, r] * C[k, r] * D[l, r]
        lambda, factors = cp_als(X, R)
        A_hat, B_hat, C_hat, D_hat = factors
        loss_hat = sqrt(cp_loss((A_hat .* lambda', B_hat, C_hat, D_hat), X)) / norm(X)
        @test isfinite(loss_hat) && loss_hat < 1e-7

        # Test cp_als on the synthetic tensor with 3-tensor
        I, J, K, R = 20, 30, 50, 10
        A, B, C = randn(I, R), randn(J, R), randn(K, R)
        @tullio X[i, j, k] := A[i, r] * B[j, r] * C[k, r]

        # Test cp_loss with the true factors
        loss = sqrt(cp_loss((A, B, C), X)) / norm(X)      
        @test isfinite(loss) && loss < 1e-7

        # Test cp_loss with the MTTKRP for mode 1
        Gts = (A' * A, B' * B, C' * C)
        X1 = reshape(X, I, J*K)
        KR = Matrix{Float64}(undef, J*K, R)
        khatri_rao!(KR, C, B)
        mttkrp_n = X1 * KR # MTTKRP for mode 1
        X_norm2 = sum(abs2, X)
        loss2 = sqrt(cp_loss(Gts, A, mttkrp_n, X_norm2)) / norm(X)
        @test isfinite(loss2) && loss2 < 1e-7

        # Test mttkrp! against the naive MTTKRP
        mttkrp_n2 = similar(mttkrp_n)
        TensorFactors.mttkrp!(mttkrp_n2, X, (A, B, C), Val(1))
        @test norm(mttkrp_n2 - mttkrp_n) / norm(mttkrp_n) < 1e-8

        # Test cp_als on the synthetic tensor
        lambda, A_hat, B_hat, C_hat = cp_als(X, R; show_trace=true, show_every=50)
        loss_hat = sqrt(cp_loss((A_hat .* lambda', B_hat, C_hat), X)) / norm(X)
        println("Loss after ALS optimization: $loss_hat")
        @test isfinite(loss_hat) && loss_hat < 1e-7

        # Test conversion between flat parameter vector and CP factors
        p = TensorFactors.cp_factors_to_flat((A, B, C))
        A_hat, B_hat, C_hat = TensorFactors.flat_to_cp_factors(p, R, size(X))
        @test (norm(A - A_hat) / norm(A) < 1e-14 || norm(B - B_hat) / norm(B) < 1e-14
            || norm(C - C_hat) / norm(C) < 1e-14)

        # Test flat loss
        norm2_X = sum(abs2, X)
        loss = cp_loss(p, X, R, norm2_X)
        @test isfinite(loss) && loss < 1e-7

        # Test flat gradient
        p = randn(length(p))
        g = similar(p)
        TensorFactors.cp_loss_grad!(g, p, X, R)
        g_fd = similar(p)
        ForwardDiff.gradient!(g_fd, p -> cp_loss(p, X, R, norm2_X), p)
        @test norm(g - g_fd) / norm(g_fd) < 1e-7

        # Test optimization based CPD with random initialization
        A_hat, B_hat, C_hat = cp_opt(ConjugateGradient(), X, R; show_trace=true, show_every=50)
        loss_hat = sqrt(cp_loss((A_hat, B_hat, C_hat), X, norm2_X)) / norm(X)
        println("Loss after optimization with random initialization: $loss_hat")
        @test isfinite(loss_hat) && loss_hat < 1e-12

        # # Test optimization based CPD with ALS initialization
        # lambda, A_hat, B_hat, C_hat = cp_als(X, R; dloss_rtol=1e-7, show_trace=true, show_every=50)
        # A_hat, B_hat, C_hat = cp_opt(ConjugateGradient(), X, R; show_trace=true, show_every=1, init_factors=(A_hat .* lambda', B_hat, C_hat))
        # loss_hat = sqrt(cp_loss((A_hat, B_hat, C_hat), X, norm2_X)) / norm(X)
        # println("Loss after optimization with ALS initialization: $loss_hat")
        # @test isfinite(loss_hat) && loss_hat < 1e-12
    end
end

function main()
    @testset "TensorFactors.jl" begin
        test_cp()
    end
end

main()