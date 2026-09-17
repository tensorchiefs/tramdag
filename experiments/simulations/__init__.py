"""The SCM generators of the paper's experiments.

Research code, not framework code: it lives with the experiments that
consume it. Each scenario is one module exposing a numpy-only SCM generator class with
known causal ground truth, imported from that module by name
(``from simulations.triangle import TriangleContinuous``). Frozen CSVs
live under ``data/<name>/`` and are a contract — regenerate only deliberately
via each module's CLI.
"""
