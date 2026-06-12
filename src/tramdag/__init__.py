"""tramdag — Interpretable Neural Causal Models (TRAM-DAGs) in PyTorch.

One triangular normalizing flow (built on `zuko <https://zuko.readthedocs.io>`_)
whose Jacobian sparsity is the causal DAG: fit once on observational data,
then answer observational (L1), interventional (L2) and counterfactual (L3)
queries. Reference: Sick & Dürr, *Interpretable Neural Causal Models with
TRAM-DAGs*, CLeaR 2025 (arXiv:2503.16206).

Conventional import alias::

    import tramdag as td

    flow = td.CausalFlowDAG(spec)
    td.simulations.REGISTRY          # synthetic DGPs with known ground truth
"""

from . import simulations
from .flow import CausalFlowDAG
from .spec import ContinuousNode, OrdinalNode

__all__ = ["CausalFlowDAG", "ContinuousNode", "OrdinalNode", "simulations"]
__version__ = "0.2.0"
