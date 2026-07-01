using Test
using QuantumNaturalfPEPS

@testset "QuantumNaturalfPEPS tests" begin
    # run every test file in the \Tests directory
    for file in sort(readdir(@__DIR__))
        if file != "runtests.jl" && endswith(file, ".jl")
            @testset "$file" begin
                @info "Running test file: $file"
                include(joinpath(@__DIR__, file))
            end
        end
    end
end;