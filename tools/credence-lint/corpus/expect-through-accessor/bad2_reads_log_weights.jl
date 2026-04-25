# Role: brain-side application
# Reads log_weights directly — should use weights(m).

using Credence

function get_probs(m)
    lw = m.log_weights
    exp.(lw .- maximum(lw))
end
