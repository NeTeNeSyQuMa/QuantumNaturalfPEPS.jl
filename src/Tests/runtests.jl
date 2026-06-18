using Test
using QuantumNaturalfPEPS
using Logging

# This sets the logging level to only show errors and above, effectively suppressing warnings.
struct NoWarnLogger <: AbstractLogger
    parent::AbstractLogger
end

Logging.min_enabled_level(::NoWarnLogger) = Logging.Info
Logging.catch_exceptions(logger::NoWarnLogger) = Logging.catch_exceptions(logger.parent)
Logging.shouldlog(logger::NoWarnLogger, level, _module, group, id) = level != Logging.Warn && Logging.shouldlog(logger.parent, level, _module, group, id)
Logging.handle_message(logger::NoWarnLogger, args...) = Logging.handle_message(logger.parent, args...)

global_logger(NoWarnLogger(ConsoleLogger(stderr, Logging.Info)))

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