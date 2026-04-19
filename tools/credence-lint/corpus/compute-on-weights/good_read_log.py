# Role: body
"""Reading weights for diagnostic logging is sanctioned access."""
import logging
from skin.client import SkinClient

log = logging.getLogger(__name__)


def log_posterior(skin: SkinClient, state_id: str) -> None:
    w = skin.weights(state_id)
    log.info("posterior weights: %s", w)
