"""
    preferences.jl — Email action space and outcome classification

Defines the action vocabulary for the email domain.
"""

const EMAIL_ACTIONS = [:archive, :flag_urgent, :schedule_later,
                       :draft_response, :delegate, :summarise, :ask_user]

const DOMAIN_ACTIONS = [:archive, :flag_urgent, :schedule_later,
                        :draft_response, :delegate, :summarise]

# Meta-actions: computational operations evaluated by EU
const META_ACTIONS = [:enumerate_more, :perturb_grammar, :deepen, :do_nothing]
const ALL_ACTIONS = vcat(EMAIL_ACTIONS, META_ACTIONS)

# Costs: fraction of approval utility lost by spending a turn computing
const ENUMERATE_COST = 0.05
const PERTURB_COST = 0.05
const DEEPEN_COST = 0.10
