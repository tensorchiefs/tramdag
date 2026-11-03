"""Interpretable Neural Causal Models (TRAM-DAGs) in PyTorch."""

# %% imports ---------------------------------------------------------------------------
from importlib.metadata import version as _version

from .callbacks import EarlyStopping
from .flow import CausalFlowDAG
from .nodes import Node
from .plots import plot_dag, plot_marginals, plot_training, plot_varying_coef
from .spec import (
    CI,
    CS,
    LS,
    SI,
    VC,
    ComplexShift,
    ContinuousNode,
    I,
    Intercept,
    LinearShift,
    OrdinalNode,
    Term,
    VaryingCoefficient,
    node_parents,
    spec_from_dict,
    spec_to_dict,
    validate_and_sort,
)

# %% global variables ------------------------------------------------------------------
__all__ = [
    "CI",
    "CS",
    "LS",
    "SI",
    "VC",
    "CausalFlowDAG",
    "ComplexShift",
    "ContinuousNode",
    "EarlyStopping",
    "I",
    "Intercept",
    "LinearShift",
    "Node",
    "OrdinalNode",
    "Term",
    "VaryingCoefficient",
    "node_parents",
    "plot_dag",
    "plot_marginals",
    "plot_training",
    "plot_varying_coef",
    "spec_from_dict",
    "spec_to_dict",
    "validate_and_sort",
]
__version__ = _version("tramdag")
