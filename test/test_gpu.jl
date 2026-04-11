using TensorFactors
using LinearAlgebra
using CUDA
using Tullio, KernelAbstractions
using Random
using ForwardDiff
using Optim
using Test

function test_cp()
    function khatri_rao!(KR::CuArray{T,2}, C::CuArray{T,2}, B::CuArray{T,2}) where {T}
        K, R  = size(C)
        J, R2 = size(B)
        @assert R == R2
        @assert size(KR) == (J*K, R)

        KR .= reshape(reshape(B, J, 1, R) .* reshape(C, 1, K, R), J*K, R)
        return KR
    end

    @testset "cp.jl" begin
        Random.seed!(42)
        CUDA.seed!(42)

        # Test column normalization
        A = CuArray(randn(20, 10))
        TensorFactors.colnormalize!(A)
        all_norms = vec(norm.(eachcol(A)))
        @test all(isfinite.(all_norms)) && all(isapprox.(all_norms, 1.0; rtol=1e-12))
        
        # Test column normalization with lambda obsorbed
        A = CuArray(randn(20, 10))
        A_copy = copy(A)
        lambda = CuArray(Vector{Float64}(undef, size(A, 2)))
        TensorFactors.colnormalize!(A, lambda)
        all_norms = vec(norm.(eachcol(A)))
        A_hat = A .* lambda'
        @test all(isapprox.(all_norms, 1.0; rtol=1e-12))
        @test norm(all_norms .- 1.0) / sqrt(length(all_norms)) < 1e-12
        @test norm(A_hat - A_copy) / norm(A_copy) < 1e-12

        # # Test cp_als on the synthetic tensor with 4-tensor
        # I, J, K, L, R = 10, 20, 30, 40, 10
        # A, B, C, D = randn(I, R), randn(J, R), randn(K, R), randn(L, R)
        # X = zeros(I, J, K, L)
        # @tullio X[i, j, k, l] := A[i, r] * B[j, r] * C[k, r] * D[l, r]
        # norm2_X = norm(X)^2
        # lambda, factors = cp_als(X, R)
        # A_hat, B_hat, C_hat, D_hat = factors
        # loss_hat = sqrt(cp_loss((A_hat .* lambda', B_hat, C_hat, D_hat), X, norm2_X)) / norm(X)
        # @test loss_hat < 1e-6

        # Test cp_als on the synthetic tensor with 3-tensor
        I, J, K, R = 20, 30, 40, 10
        A, B, C = CuArray(randn(I, R)), CuArray(randn(J, R)), CuArray(randn(K, R))
        @tullio X[i, j, k] := A[i, r] * B[j, r] * C[k, r]
        norm2_X = norm(X)^2

        # Test contraction
        X_hat = cp_contract((A, B, C))
        rel_err = norm(X - X_hat) / norm(X)
        @test rel_err < 1e-12

        # Test cp_loss with the true factors
        loss = sqrt(cp_loss((A, B, C), X, norm2_X)) / norm(X)
        @test loss < 1e-12

        # Test cp_loss with the MTTKRP for mode 1
        GtA = A' * A
        GtB = B' * B
        GtC = C' * C
        X1 = reshape(X, I, J*K)
        KR = CuArray{Float64}(undef, J*K, R)
        khatri_rao!(KR, C, B)
        mttkrp_n = X1 * KR # MTTKRP for mode 1
        loss2 = sqrt(cp_loss((GtA, GtB, GtC), A, mttkrp_n, norm2_X)) / norm(X)
        @test loss2 < 1e-10

        # Test mttkrp! against the naive MTTKRP
        # mttkrp_n2 = similar(mttkrp_n)
        # TensorFactors.mttkrp!(mttkrp_n2, X, (A, B, C), Val(1))
        # @test norm(mttkrp_n2 - mttkrp_n) / norm(mttkrp_n) < 1e-10

        # Test cp_als on the synthetic tensor
        lambda, A_hat, B_hat, C_hat = cp_als(X, R; show_trace=true, show_every=50)
        loss_hat = sqrt(cp_loss((A_hat .* lambda', B_hat, C_hat), X, norm2_X)) / norm(X)
        println("Loss after ALS optimization: $loss_hat")
        @test loss_hat < 1e-6

        # Test conversion between flat parameter vector and CP factors
        p = TensorFactors.cp_factors_to_flat((A, B, C))
        A_hat, B_hat, C_hat = TensorFactors.flat_to_cp_factors(p, R, size(X))
        @test (norm(A - A_hat) / norm(A) < 1e-14 || norm(B - B_hat) / norm(B) < 1e-14
            || norm(C - C_hat) / norm(C) < 1e-14)

        # Test optimization based CPD with random initialization
        A_hat, B_hat, C_hat = cp_opt(ConjugateGradient(), X, R; show_trace=true, show_every=100)
        loss_hat = sqrt(cp_loss((A_hat, B_hat, C_hat), X, norm2_X)) / norm(X)
        println("Loss after optimization with random initialization: $loss_hat")
        @test loss_hat < 1e-6

        # Test optimization based CPD with ALS initialization
        lambda, A_hat, B_hat, C_hat = cp_als(X, R; dloss_rtol=1e-7, show_trace=true, show_every=10)
        A_hat, B_hat, C_hat = cp_opt(ConjugateGradient(), X, R; show_trace=true, show_every=1, init_factors=(A_hat .* lambda', B_hat, C_hat))
        loss_hat = sqrt(cp_loss((A_hat, B_hat, C_hat), X, norm2_X)) / norm(X)
        println("Loss after optimization with ALS initialization: $loss_hat")
        @test loss_hat < 1e-6
    end
end

function test_tucker()
    @testset "tucker.jl" begin
        Random.seed!(42)
        CUDA.seed!(42)

        # Tensor times matrix (TTM) test
        A = CuArray(randn(4, 5, 6))
        M = CuArray(randn(3, 4))
        mode = 1
        res = ttm(A, M, mode)

        res_naive = CUDA.zeros(3, 5, 6)
        @tullio res_naive[i, j, k] := M[i, m] * A[m, j, k]
        rel_err = norm(res - res_naive) / norm(res_naive)
        @test rel_err < 1e-14

        # Contraction test
        I, J, K = 20, 30, 40
        S_I, S_J, S_K = 10, 20, 30
        U1, U2, U3 = CuArray(randn(I, S_I)), CuArray(randn(J, S_J)), CuArray(randn(K, S_K))
        S = CuArray(randn(S_I, S_J, S_K))
        A = similar(S)
        @tullio A[i, j, k] := S[a, b, c] * U1[i, a] * U2[j, b] * U3[k, c]
        A_hat = tucker_contract(S, (U1, U2, U3))
        rel_err = norm(A - A_hat) / norm(A)
        @test rel_err < 1e-14

        # HOSVD test
        S_approx, U = tucker_hosvd(A, size(S))
        U1_approx, U2_approx, U3_approx = U
        A_approx = similar(A)
        @tullio A_approx[i, j, k] := S_approx[a, b, c] * U1_approx[i, a] * U2_approx[j, b] * U3_approx[k, c]
        rel_err = norm(A - A_approx) / norm(A)
        println("Relative error after HOSVD: $rel_err")
        @test rel_err < 1e-12
    end
end