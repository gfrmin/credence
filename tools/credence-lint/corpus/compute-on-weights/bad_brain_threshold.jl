# Role: brain
# The Route-B guardrail: extracting the model-averaged predictive and choosing
# the action by a host-side threshold is a parallel decision mechanism. Hand the
# belief + a typed cost Functional to `optimise` instead. The thin DSL surface
# used to make this UNSAYABLE; in typed Julia the brain it must be lint-caught.

using Credence

function decide(belief_at_X)
    p = expect(belief_at_X, Identity())       # P(approve | X)
    return p < 0.5 ? :block : :proceed        # violation: host-side argmax on a belief scalar
end
