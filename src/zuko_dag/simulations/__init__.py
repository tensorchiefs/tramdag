"""Synthetic-cohort generators for zuko_dag.

Each scenario is one module exposing a generator class. New scenarios register
here so experiments/tests can look them up by name.
"""

from .magic_mrclean import MagicMrClean

REGISTRY = {"magic-mrclean": MagicMrClean}

__all__ = ["MagicMrClean", "REGISTRY"]
