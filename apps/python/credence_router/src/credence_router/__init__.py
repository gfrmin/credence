# Role: body
"""credence-router: transparent, cost-optimal tool routing via EU maximisation."""

from credence_router.answer import Answer
from credence_router.config import RoutingParams, derive_routing_params
from credence_router.group import RouterGroup
from credence_router.router import Router
from credence_router.tool import Tool
from credence_router.tools.coverage_prior import CoveragePrior
from credence_router.tools.keyword_category import KeywordCategoryTool

__all__ = [
    "Answer",
    "CoveragePrior",
    "KeywordCategoryTool",
    "Router",
    "RouterGroup",
    "RoutingParams",
    "Tool",
    "create_react_agent",
    "derive_routing_params",
]


def create_react_agent(*args, **kwargs):
    """LangGraph-compatible create_react_agent with VOI-based routing.

    Requires: pip install credence-router[compat]
    """
    from credence_router.compat.prebuilt import create_react_agent as _cra

    return _cra(*args, **kwargs)
