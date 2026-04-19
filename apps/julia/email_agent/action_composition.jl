# Role: brain-side application
"""
    action_composition.jl — Host-level action decomposition (spec §7.4)

Programs output single action Symbols. The host decomposes composite
actions into sequences of low-level primitives for execution. This
draws the body boundary at the composite action level — the brain
(programs) decides what to do, the body (host) expands how.

The agent discovers which granularity of action the user cares about
through Bayesian conditioning. If the user distinguishes between
"archive with notification" and "archive silently," programs recommending
the correct composite gain weight. If they don't, simpler actions
dominate via Occam's Razor.
"""

const LOW_LEVEL_PRIMITIVES = [
    :move_to_archive, :move_to_priority, :move_to_later,
    :add_label_urgent, :add_label_delegated,
    :mark_read, :notify_user, :draft_reply, :assign_to
]

const ACTION_COMPOSITIONS = Dict{Symbol, Vector{Symbol}}(
    :archive         => [:mark_read, :move_to_archive],
    :flag_urgent     => [:add_label_urgent, :move_to_priority, :notify_user],
    :schedule_later  => [:move_to_later],
    :draft_response  => [:draft_reply, :notify_user],
    :delegate        => [:add_label_delegated, :assign_to],
    :summarise       => [:mark_read],
    # New composites — the body expands its action vocabulary
    :triage_urgent   => [:add_label_urgent, :move_to_priority, :notify_user, :assign_to],
    :silent_archive  => [:mark_read, :move_to_archive],
    :escalate        => [:add_label_urgent, :move_to_priority, :notify_user],
)

"""
    decompose_action(action) → Vector{Symbol}

Decompose a composite action into its low-level primitives.
Returns [action] if no composition is defined (pass-through).
"""
function decompose_action(action::Symbol)::Vector{Symbol}
    get(ACTION_COMPOSITIONS, action, [action])
end
