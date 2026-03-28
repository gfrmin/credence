"""
    preferences.jl — Email action space and outcome classification

Defines the action vocabulary for the email domain.
"""

const EMAIL_ACTIONS = [:archive, :flag_urgent, :schedule_later,
                       :draft_response, :delegate, :summarise, :ask_user]

const DOMAIN_ACTIONS = [:archive, :flag_urgent, :schedule_later,
                        :draft_response, :delegate, :summarise,
                        :triage_urgent, :silent_archive, :escalate]

# Meta-actions: computational operations evaluated by EU
const META_ACTIONS = [:enumerate_more, :perturb_grammar, :deepen, :do_nothing]

# Sensor actions: gather more information before deciding
const SENSOR_ACTIONS = [:ask_llm]

const ALL_ACTIONS = vcat(EMAIL_ACTIONS, META_ACTIONS, SENSOR_ACTIONS)

# Primitive actions for multi-step episodes (spec §7.4)
const PRIMITIVE_ACTIONS = [
    :add_label_urgent, :add_label_delegated,
    :move_to_archive, :move_to_priority, :move_to_later,
    :mark_read, :notify_user, :draft_reply, :assign_to, :done
]

# Target state: each composite action decomposes into a set of primitives
const ACTION_TARGET_STATE = Dict{Symbol, Set{Symbol}}(
    :archive         => Set([:mark_read, :move_to_archive]),
    :flag_urgent     => Set([:add_label_urgent, :move_to_priority, :notify_user]),
    :schedule_later  => Set([:move_to_later]),
    :draft_response  => Set([:draft_reply, :notify_user]),
    :delegate        => Set([:add_label_delegated, :assign_to]),
    :summarise       => Set([:mark_read]),
    :triage_urgent   => Set([:add_label_urgent, :move_to_priority, :notify_user, :assign_to]),
    :silent_archive  => Set([:mark_read, :move_to_archive]),
    :escalate        => Set([:add_label_urgent, :move_to_priority, :notify_user]),
)

# Combined action space for multi-step episodes
const PRIMITIVE_ALL_ACTIONS = vcat(PRIMITIVE_ACTIONS, [:ask_user], META_ACTIONS, SENSOR_ACTIONS)

# Costs: fraction of approval utility lost by spending a turn computing
const ENUMERATE_COST = 0.05
const PERTURB_COST = 0.05
const DEEPEN_COST = 0.10
