"""CausalFlowDAG — a single triangular normalizing flow on a user-defined DAG.

The flow maps iid standard-logistic latents ``U`` to the observed variables ``X``
in topological order; its Jacobian sparsity is exactly the DAG adjacency. The
joint log-likelihood decomposes per node, so one optimizer fits all nodes at once.

Causal queries:
    flow.sample(n)                    observational sampling
    flow.sample(n, do={"T": 1})       interventional sampling (graph mutilation)
    u = flow.abduct(df)               Pearl step 1 (latents from observations)
    flow.sample(do={"T": 1}, u=u)     Pearl steps 2+3 (counterfactuals)
    flow.pmf(df, node, do=...)        analytic per-row interventional PMF
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

from .conditioners import ComplexIntercept, ComplexShift, LinearShift, SimpleIntercept
from .spec import (ContinuousNode, NodeSpec, OrdinalNode, spec_from_dict,
                   spec_to_dict, validate_and_sort)
from .transforms import (StandardLogistic, make_univariate_transform,
                         ordinal_abduct, ordinal_log_prob, ordinal_pmf,
                         ordinal_sample)

__all__ = ["CausalFlowDAG"]


class _Node(nn.Module):
    """One dimension of the flow: intercept (transform params) + additive shifts."""

    def __init__(self, name: str, node: NodeSpec, spec: dict[str, NodeSpec]):
        super().__init__()
        self.name = name
        self.kind = node.kind
        self.parents = dict(node.parents)
        self.ci_parents = [p for p, t in node.parents.items() if t == "ci"]

        if isinstance(node, ContinuousNode):
            self.ut = make_univariate_transform(node.transform, **node.transform_kwargs)
            n_params = self.ut.n_params
            self.levels = None
        else:
            self.ut = None
            self.levels = node.levels
            n_params = node.levels - 1

        def width(parent: str) -> int:
            pn = spec[parent]
            return pn.levels if isinstance(pn, OrdinalNode) else 1

        if self.ci_parents:
            self.intercept = ComplexIntercept(sum(width(p) for p in self.ci_parents), n_params)
        else:
            self.intercept = SimpleIntercept(n_params)

        self.shifts = nn.ModuleDict()
        for parent, term in node.parents.items():
            if term == "ls":
                self.shifts[parent] = LinearShift(width(parent))
            elif term == "cs":
                self.shifts[parent] = ComplexShift(width(parent))

    def theta_shift(self, feats: dict[str, Tensor], n: int) -> tuple[Tensor, Tensor]:
        """Transform parameters (n, P) and total shift (n,) from parent features."""
        if self.ci_parents:
            theta = self.intercept(torch.cat([feats[p] for p in self.ci_parents], dim=1))
        else:
            theta = self.intercept(n)
        shift = torch.zeros(n, dtype=theta.dtype, device=theta.device)
        for parent, module in self.shifts.items():
            shift = shift + module(feats[parent])
        return theta, shift


class CausalFlowDAG(nn.Module):
    """A causal normalizing flow defined by ``spec = {name: NodeSpec}``."""

    def __init__(self, spec: dict[str, NodeSpec], device: str = "cpu"):
        super().__init__()
        self.spec = spec
        self.order = validate_and_sort(spec)
        self.nodes = nn.ModuleDict({name: _Node(name, spec[name], spec) for name in self.order})
        self.device = torch.device(device)
        self.history: dict = {"train": [], "val": [], "lr": [], "time": []}
        self.to(self.device)

    # ------------------------------------------------------------------ data
    def _encode_parent(self, name: str, values: Tensor) -> Tensor:
        """Encode a node's values for use as a parent feature (tramdag convention:
        continuous raw (n, 1); ordinal one-hot (n, levels))."""
        node = self.spec[name]
        if isinstance(node, OrdinalNode):
            return torch.nn.functional.one_hot(
                values.long(), num_classes=node.levels).to(values.dtype)
        return values.view(-1, 1)

    def _tensorize(self, df: pd.DataFrame) -> dict[str, Tensor]:
        out = {}
        for name in self.order:
            vals = torch.as_tensor(
                df[name].to_numpy(dtype=np.float32), device=self.device)
            out[name] = vals
        return out

    def _features(self, values: dict[str, Tensor]) -> dict[str, Tensor]:
        return {name: self._encode_parent(name, vals) for name, vals in values.items()}

    # ------------------------------------------------------------- likelihood
    def node_log_prob(self, values: dict[str, Tensor]) -> dict[str, Tensor]:
        """Per-node log-likelihood contributions, each (n,)."""
        feats = self._features(values)
        n = next(iter(values.values())).shape[0]
        out = {}
        for name in self.order:
            node = self.nodes[name]
            theta, shift = node.theta_shift(feats, n)
            x = values[name]
            if node.kind == "continuous":
                z0, ladj = node.ut.forward(theta, x)
                z = z0 + shift
                out[name] = StandardLogistic.log_prob(z) + ladj
            else:
                out[name] = ordinal_log_prob(theta, shift, x)
        return out

    def log_prob(self, df: pd.DataFrame) -> Tensor:
        """Joint log-likelihood log p(x) per row, shape (n,)."""
        per_node = self.node_log_prob(self._tensorize(df))
        return torch.stack(list(per_node.values()), dim=0).sum(dim=0)

    def nll(self, df: pd.DataFrame) -> dict[str, float]:
        """Mean negative log-likelihood per node (diagnostic)."""
        with torch.no_grad():
            per_node = self.node_log_prob(self._tensorize(df))
        return {k: float(-v.mean()) for k, v in per_node.items()}

    # ------------------------------------------------------------------- fit
    def _set_ranges(self, train_df: pd.DataFrame) -> None:
        """Train 5%/95% quantiles -> transform domain (tramdag's min_max scaling)."""
        for name in self.order:
            node = self.nodes[name]
            if node.kind == "continuous" and not node.ut._fitted:
                q = train_df[name].quantile([0.05, 0.95])
                node.ut.set_range(q.iloc[0], q.iloc[1])

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None,
            epochs: int = 500, learning_rate: float = 1e-2, batch_size: int = 512,
            verbose: int = 50, seed: int | None = None) -> "CausalFlowDAG":
        """Jointly fit all nodes by maximum likelihood.

        Mirrors tramdag's convention: per-node best-validation weights are
        tracked and restored at the end. Calling ``fit`` again continues
        training (e.g. a second phase with a lower learning rate).
        """
        if seed is not None:
            torch.manual_seed(seed)
        self._set_ranges(train_df)

        train_vals = self._tensorize(train_df)
        val_vals = self._tensorize(val_df) if val_df is not None else train_vals
        n = len(train_df)

        opt = torch.optim.Adam(self.parameters(), lr=learning_rate)
        if not hasattr(self, "_best"):
            self._best = {name: (float("inf"), None) for name in self.order}
        best = self._best
        t0 = time.perf_counter()
        t_offset = self.history["time"][-1] if self.history.get("time") else 0.0

        for epoch in range(epochs):
            self.train()
            perm = torch.randperm(n, device=self.device)
            train_acc = {name: 0.0 for name in self.order}
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                batch = {k: v[idx] for k, v in train_vals.items()}
                per_node = self.node_log_prob(batch)
                node_nlls = {k: -v.mean() for k, v in per_node.items()}
                loss = torch.stack(list(node_nlls.values())).sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
                w = len(idx) / n
                for k, v in node_nlls.items():
                    train_acc[k] += float(v.detach()) * w

            self.eval()
            with torch.no_grad():
                val_per_node = {k: float(-v.mean())
                                for k, v in self.node_log_prob(val_vals).items()}
            self.history["train"].append(train_acc)
            self.history["val"].append(val_per_node)
            self.history.setdefault("lr", []).append(learning_rate)
            self.history.setdefault("time", []).append(
                t_offset + time.perf_counter() - t0)

            for name in self.order:
                if val_per_node[name] < best[name][0]:
                    best[name] = (val_per_node[name],
                                  copy.deepcopy(self.nodes[name].state_dict()))

            if verbose and (epoch % verbose == 0 or epoch == epochs - 1):
                tot_t = sum(train_acc.values())
                tot_v = sum(val_per_node.values())
                print(f"[epoch {epoch + 1:5d}/{epochs}] train NLL {tot_t:.4f}  "
                      f"val NLL {tot_v:.4f}")

        for name, (_, state) in best.items():  # restore per-node best-val weights
            if state is not None:
                self.nodes[name].load_state_dict(state)
        self.eval()
        return self

    # ------------------------------------------------------- causal queries
    @torch.no_grad()
    def sample(self, n: int | None = None, *, do: dict[str, float] | None = None,
               u: pd.DataFrame | None = None, seed: int | None = None) -> pd.DataFrame:
        """Sample from the (optionally mutilated) flow.

        Args:
            n: number of samples (ignored if ``u`` is given).
            do: interventions {node: value}; intervened nodes are clamped and
                their parent dependence removed (graph mutilation).
            u: latent variables (as returned by :meth:`abduct`). If given, they
                are pushed through the flow — together with ``do`` this yields
                counterfactuals (Pearl's abduction -> action -> prediction).
        """
        do = do or {}
        gen = None
        if seed is not None:
            gen = torch.Generator(device=self.device).manual_seed(seed)

        if u is not None:
            n = len(u)
            u_vals = {name: torch.as_tensor(u[name].to_numpy(dtype=np.float32, copy=True),
                                            device=self.device) for name in self.order}
        elif n is not None:
            u_vals = {name: StandardLogistic.sample((n,), device=self.device)
                      if gen is None else
                      StandardLogistic.icdf(torch.rand((n,), device=self.device, generator=gen))
                      for name in self.order}
        else:
            raise ValueError("Provide either n or u.")

        values: dict[str, Tensor] = {}
        for name in self.order:
            if name in do:
                values[name] = torch.full((n,), float(do[name]), device=self.device)
                continue
            node = self.nodes[name]
            feats = self._features({p: values[p] for p in node.parents})
            theta, shift = node.theta_shift(feats, n)
            z = u_vals[name]
            if node.kind == "continuous":
                values[name] = node.ut.inverse(theta, z - shift)
            else:
                values[name] = ordinal_sample(theta, shift, z)
        return pd.DataFrame({k: v.cpu().numpy() for k, v in values.items()})

    @torch.no_grad()
    def abduct(self, df: pd.DataFrame, seed: int | None = None) -> pd.DataFrame:
        """Pearl abduction: recover the latent variables ``u`` from observations.

        Continuous nodes are inverted exactly (``u = h(x) + shift``); for ordinal
        nodes the latent is only interval-identified, so it is sampled from the
        standard logistic truncated to the observed level's interval.
        """
        gen = None
        if seed is not None:
            gen = torch.Generator(device=self.device).manual_seed(seed)
        values = self._tensorize(df)
        feats = self._features(values)
        n = len(df)
        u = {}
        for name in self.order:
            node = self.nodes[name]
            theta, shift = node.theta_shift(feats, n)
            x = values[name]
            if node.kind == "continuous":
                z0, _ = node.ut.forward(theta, x)
                u[name] = z0 + shift
            else:
                u[name] = ordinal_abduct(theta, shift, x, generator=gen)
        return pd.DataFrame({k: v.cpu().numpy() for k, v in u.items()})

    @torch.no_grad()
    def pmf(self, df: pd.DataFrame, node: str, do: dict[str, float] | None = None) -> np.ndarray:
        """Analytic class probabilities (n, levels) for an ordinal node, with the
        node's parents taken from ``df`` after applying ``do`` overrides."""
        if not isinstance(self.spec[node], OrdinalNode):
            raise ValueError(f"pmf() requires an ordinal node, '{node}' is continuous.")
        df_local = df.copy()
        for col, val in (do or {}).items():
            df_local[col] = val
        nd = self.nodes[node]
        values = {p: torch.as_tensor(df_local[p].to_numpy(dtype=np.float32),
                                     device=self.device) for p in nd.parents}
        feats = self._features(values)
        theta, shift = nd.theta_shift(feats, len(df_local))
        return ordinal_pmf(theta, shift).cpu().numpy()

    # ------------------------------------------------------------------- io
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"spec": spec_to_dict(self.spec),
                    "state_dict": self.state_dict(),
                    "history": self.history}, path)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "CausalFlowDAG":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        flow = cls(spec_from_dict(ckpt["spec"]), device=device)
        for name in flow.order:  # mark transforms as fitted before loading buffers
            node = flow.nodes[name]
            if node.kind == "continuous":
                node.ut._fitted = True
        flow.load_state_dict(ckpt["state_dict"])
        flow.history = ckpt.get("history", {"train": [], "val": []})
        flow.eval()
        return flow
