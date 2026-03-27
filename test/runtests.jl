using TensorFactors
using LinearAlgebra
using LoopVectorization
using Tullio
using Random
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
        A_hat, B_hat, C_hat = cp_als(X, R)
        loss_hat = sqrt(cp_loss((A_hat, B_hat, C_hat), X)) / norm(X)
        @test isfinite(loss_hat) && loss_hat < 1e-7

        # Test cp_als on the synthetic tensor with more modes
        I, J, K, L, R = 20, 30, 50, 60, 10
        A, B, C, D = randn(I, R), randn(J, R), randn(K, R), randn(L, R)
        X = zeros(I, J, K, L)
        @tullio X[i, j, k, l] := A[i, r] * B[j, r] * C[k, r] * D[l, r]
        A_hat, B_hat, C_hat, D_hat = cp_als(X, R)
        loss_hat = sqrt(cp_loss((A_hat, B_hat, C_hat, D_hat), X)) / norm(X)
        @test isfinite(loss_hat) && loss_hat < 1e-7
    end
end

function main()
    @testset "TensorFactors.jl" begin
        test_cp()
    end
end

main()