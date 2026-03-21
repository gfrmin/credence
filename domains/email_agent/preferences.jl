"""
    preferences.jl — Email action space and outcome classification

Defines the action vocabulary for the email domain.
"""

const EMAIL_ACTIONS = [:archive, :flag_urgent, :schedule_later,
                       :draft_response, :delegate, :summarise, :ask_user]

const DOMAIN_ACTIONS = [:archive, :flag_urgent, :schedule_later,
                        :draft_response, :delegate, :summarise]
