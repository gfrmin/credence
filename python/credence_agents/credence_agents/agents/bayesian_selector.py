"""BayesianSelector: base class for EU-maximising selection over options.

Options can be tools (BayesianAgent), providers (RoutingDomain),
models, or anything with per-category reliability. All inference
runs in the Julia Credence DSL via CredenceBridge.

This class owns the shared machinery:
- Julia state (rel_states, cov_states, cat_belief)
- EU maximisation via agent-step
- Reliability updates via update_beta_state
- State introspection (learned_reliability)

Subclasses add domain-specific logic:
- BayesianAgent: MCQ answer measures, tool responses, submit/abstain
- RoutingDomain: provider forwarding, streaming, observation extraction
"""

from __future__ import annotations

from credence_agents.inference.voi import ScoringRule, ToolConfig
from credence_agents.julia_bridge import CredenceBridge


class BayesianSelector:
    """EU-maximising selection over options with per-category reliability.

    All Bayesian computation runs in the Julia Credence DSL.
    """

    def __init__(
        self,
        bridge: CredenceBridge,
        option_configs: list[ToolConfig],
        categories: tuple[str, ...],
        scoring: ScoringRule = ScoringRule(),
    ):
        self.bridge = bridge
        self.option_configs = option_configs
        self.scoring = scoring
        self.n_options = len(option_configs)
        self._n_cats = len(categories)
        self._categories = categories

        # Julia state: per-option MixtureMeasures + category belief
        self.rel_states = [bridge.initial_rel_state(self._n_cats) for _ in option_configs]
        self.cov_states = [
            bridge.initial_cov_state(self._n_cats, oc.coverage_by_category)
            for oc in option_configs
        ]
        self.cat_belief = bridge.make_cat_belief(self._n_cats)

    def select(self, cat_weights: list[float] | None = None) -> int:
        """EU-maximise over options and return index of best.

        Args:
            cat_weights: category distribution for this query. If None,
                uses current cat_belief from Julia state.

        Returns:
            Index into option_configs of the selected option.
        """
        bridge = self.bridge

        if cat_weights is None:
            cat_weights = bridge.weights(self.cat_belief)

        # Build per-option reliability and coverage for available options
        all_indices = list(range(self.n_options))
        rel_measures = [
            bridge.marginalize_betas(self.rel_states[i], cat_weights)
            for i in all_indices
        ]
        cov_probs = [
            bridge.expect_identity(bridge.marginalize_betas(self.cov_states[i], cat_weights))
            for i in all_indices
        ]
        costs = [self.option_configs[i].cost for i in all_indices]

        # Temporary answer measure (2 options: "good" and "bad")
        # We only care about the query action — submit/abstain are irrelevant for selection
        answer_measure = bridge.make_answer_measure(2)

        action_type, action_arg = bridge.agent_step(
            answer_measure,
            rel_measures,
            costs,
            cov_probs,
            self.scoring.reward_correct,
            self.scoring.reward_abstain,
            self.scoring.penalty_wrong,
        )

        # agent-step returns: 0=submit, 1=abstain, 2=query
        if action_type == 2:
            return action_arg
        # If agent wants to submit/abstain (all options look bad),
        # fall back to the option with lowest effective cost
        return min(all_indices, key=lambda i: self.option_configs[i].cost)

    def update_reliability(self, option_idx: int, useful: float) -> None:
        """Update beliefs for the selected option from outcome.

        Args:
            option_idx: which option was used
            useful: 1.0 for good outcome, 0.0 for bad
        """
        self.rel_states[option_idx], self.cat_belief = self.bridge.update_beta_state(
            self.rel_states[option_idx],
            self.cat_belief,
            useful,
        )

    def update_coverage(self, option_idx: int, responded: float) -> None:
        """Update coverage beliefs for the selected option.

        Args:
            option_idx: which option was used
            responded: 1.0 if option returned a result, 0.0 if it didn't
        """
        self.cov_states[option_idx], self.cat_belief = self.bridge.update_beta_state(
            self.cov_states[option_idx],
            self.cat_belief,
            responded,
        )

    @property
    def learned_reliability(self) -> dict[int, list[float]]:
        """Per-option per-category reliability means.

        Returns:
            {option_idx: [reliability_per_category]}
        """
        result = {}
        for i in range(self.n_options):
            result[i] = self.bridge.extract_reliability_means(self.rel_states[i])
        return result

    @property
    def categories(self) -> tuple[str, ...]:
        return self._categories
