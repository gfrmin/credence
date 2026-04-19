# Role: brain-side application
"""
    preferences.jl — RSS action space for program enumeration.

    Programs predict :match (article is interesting) or :no_match (no opinion).
    The kernel interprets :match as "fires" and :no_match as "doesn't fire."
"""

const RSS_ACTION_SPACE = [:match, :no_match]
