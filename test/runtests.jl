using SafeTestsets

@safetestset "CPU Tests" begin
    include("test_cpu.jl")
    test_cp()
    test_tucker()
end

CUDA_available = true
GPU_available = true
try
    using CUDA
    if !CUDA.functional()
        global GPU_available = false
    end
catch
    global CUDA_available = false
end

if CUDA_available
    if GPU_available
        @safetestset "GPU Tests" begin
            include("test_gpu.jl")
            test_cp()
            #test_tucker_gpu()
        end
    else
        @warn "No functional GPU device available, skipping GPU tests"
    end
else
    @warn "CUDA not available, skipping GPU tests"
end