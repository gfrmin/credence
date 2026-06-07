# Role: eval
#
# brain_env.jl — shared loader that builds the credence-pi brain's BDSL env
# exactly as the daemon does (stdlib + declared bdsl + wire_brain!), for the
# replay/cross-eval harness. `include`d by replay.jl and crosseval.jl so both
# drive the SAME wired env closures the daemon uses. (Plain comments, not a
# docstring: a `"""..."""` here would attach to the `using` below and error.)
using Credence: Eval, Parse

include(joinpath(@__DIR__, "..", "brain", "feature_brain.jl"))
using .FeatureBrain: wire_brain!

function build_env()
    env = Eval.default_env()
    env[:__toplevel__] = true
    stdlib = joinpath(@__DIR__, "..", "..", "..", "src", "stdlib.bdsl")
    for expr in Parse.parse_all(read(stdlib, String))
        Eval.eval_dsl(expr, env)
    end
    bdsl = joinpath(@__DIR__, "..", "bdsl")
    for f in ("capabilities.bdsl", "features.bdsl", "utility.bdsl")
        for expr in Parse.parse_all(read(joinpath(bdsl, f), String))
            Eval.eval_dsl(expr, env)
        end
    end
    wire_brain!(env)
    env
end
