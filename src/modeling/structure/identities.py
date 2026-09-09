"""Data structures describing exact algebraic relations discovered between targets.

An *identity* states that one target column can be reconstructed in closed form
from other target columns and/or the input features.  Identities are discovered
from data by :mod:`modeling.structure.discovery`; they are validated against a
strict residual tolerance so that a reconstructed target is exact to within the
numerical precision of the source data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class IdentityQuality:
    """Residual diagnostics for a candidate identity."""

    r2: float
    max_abs_relative_error: float
    mean_abs_relative_error: float

    def is_exact(self, tolerance: float) -> bool:
        return self.max_abs_relative_error <= tolerance

    def to_dict(self) -> dict[str, float]:
        return {
            "r2": self.r2,
            "max_abs_relative_error": self.max_abs_relative_error,
            "mean_abs_relative_error": self.mean_abs_relative_error,
        }


@dataclass
class TargetIdentity:
    """An exact closed-form reconstruction of one target.

    Attributes
    ----------
    target:
        Name of the target this identity reconstructs.
    source_targets:
        Names of the other target columns the reconstruction consumes.
    source_features:
        Names of the feature columns the reconstruction consumes.
    kind:
        Short machine-readable label of the functional family, for example
        "proportional", "affine" or "product_polynomial".
    expression:
        Human-readable formula, suitable for a paper.
    quality:
        Residual diagnostics measured on the data the identity was fitted to.
    parameters:
        Fitted coefficients, kept for reporting and reproducibility.
    evaluate:
        Callable taking (target_values, feature_values) and returning an array,
        where both arguments map a column name to a 1-D array.
    """

    target: str
    source_targets: tuple[str, ...]
    source_features: tuple[str, ...]
    kind: str
    expression: str
    quality: IdentityQuality
    parameters: dict[str, Any] = field(default_factory=dict)
    evaluate: Callable[[dict[str, np.ndarray], dict[str, np.ndarray]], np.ndarray] | None = None

    def __call__(
        self,
        target_values: dict[str, np.ndarray],
        feature_values: dict[str, np.ndarray],
    ) -> np.ndarray:
        if self.evaluate is None:
            raise RuntimeError(f"Identity for {self.target!r} has no evaluator attached.")
        return self.evaluate(target_values, feature_values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "source_targets": list(self.source_targets),
            "source_features": list(self.source_features),
            "kind": self.kind,
            "expression": self.expression,
            "quality": self.quality.to_dict(),
            "parameters": jsonable(self.parameters),
        }


@dataclass
class TargetStructure:
    """The outcome of target-space structure discovery.

    ``independent_targets`` must be learnt by a regression model.  Every target
    in ``derived_targets`` is reconstructed analytically, in the order given by
    :meth:`reconstruction_order`, from the independent targets and the features.
    """

    target_names: tuple[str, ...]
    independent_targets: tuple[str, ...]
    identities: dict[str, TargetIdentity] = field(default_factory=dict)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    tolerance: float = 1e-6

    @property
    def derived_targets(self) -> tuple[str, ...]:
        return tuple(t for t in self.target_names if t in self.identities)

    @property
    def reduction_ratio(self) -> float:
        """Fraction of targets removed from the learning problem."""
        if not self.target_names:
            return 0.0
        return 1.0 - len(self.independent_targets) / len(self.target_names)

    def reconstruction_order(self) -> list[str]:
        """Topologically order derived targets so dependencies resolve first."""
        resolved = list(self.independent_targets)
        pending = [t for t in self.target_names if t in self.identities]
        order: list[str] = []
        # Each pass must resolve at least one target, otherwise the graph is cyclic.
        while pending:
            progressed = False
            for name in list(pending):
                identity = self.identities[name]
                if all(src in resolved for src in identity.source_targets):
                    order.append(name)
                    resolved.append(name)
                    pending.remove(name)
                    progressed = True
            if not progressed:
                raise RuntimeError(
                    f"Cyclic target identities; cannot resolve {pending!r}."
                )
        return order

    def reconstruct(
        self,
        independent_values: dict[str, np.ndarray],
        feature_values: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Expand predictions of the independent targets to all targets."""
        known = dict(independent_values)
        for name in self.reconstruction_order():
            known[name] = self.identities[name](known, feature_values)
        return {name: known[name] for name in self.target_names}

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_names": list(self.target_names),
            "independent_targets": list(self.independent_targets),
            "derived_targets": list(self.derived_targets),
            "reduction_ratio": self.reduction_ratio,
            "tolerance": self.tolerance,
            "identities": {k: v.to_dict() for k, v in self.identities.items()},
            "rejected_candidates": self.rejected,
        }


def relative_error_quality(
    actual: np.ndarray,
    reconstructed: np.ndarray,
) -> IdentityQuality:
    """Score a reconstruction by R-squared and relative error."""
    actual = np.asarray(actual, dtype=np.float64)
    reconstructed = np.asarray(reconstructed, dtype=np.float64)
    finite = np.isfinite(actual) & np.isfinite(reconstructed)
    if not np.any(finite):
        return IdentityQuality(-np.inf, np.inf, np.inf)

    a = actual[finite]
    r = reconstructed[finite]
    denominator = float(np.sum((a - a.mean()) ** 2))
    residual = float(np.sum((a - r) ** 2))
    r2 = 1.0 - residual / denominator if denominator > 0 else -np.inf

    scale = np.maximum(np.abs(a), np.finfo(np.float64).tiny)
    rel = np.abs(a - r) / scale
    return IdentityQuality(r2, float(rel.max()), float(rel.mean()))


def jsonable(value: Any) -> Any:
    """Convert numpy scalars and arrays to plain Python for JSON export."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def format_polynomial(coefficients: Sequence[float], variable: str) -> str:
    """Render polynomial coefficients (highest power first) as a formula string."""
    degree = len(coefficients) - 1
    parts: list[str] = []
    for power, coefficient in zip(range(degree, -1, -1), coefficients):
        if power == 0:
            parts.append(f"{coefficient:.9g}")
        elif power == 1:
            parts.append(f"{coefficient:.9g}*{variable}")
        else:
            parts.append(f"{coefficient:.9g}*{variable}**{power}")
    return " + ".join(parts)
