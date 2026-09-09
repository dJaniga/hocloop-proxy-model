"""Response and feature transforms for the proxy model.

Simulator responses in this setting span orders of magnitude and are strongly
right skewed, so a squared-error fit on the raw scale is dominated by a handful
of large samples and effectively ignores the rest of the design.  Fitting on a
log scale fixes that, at the cost of a back-transform bias: ``exp`` of the mean
log is the geometric, not the arithmetic, mean.  Duan's smearing estimator
corrects the bias non-parametrically from the training residuals.

Combining the log transform with the Buckingham-Pi response group gives the
transform actually used by the pipeline: divide out the physical power law,
then take logs.  What the regressor sees is the dimensionless departure from
that power law, which is both easier to fit and unit invariant.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from modeling.features.dimensional import PiBasis

logger = logging.getLogger(__name__)


class ResponseTransform(ABC):
    """Invertible map between a physical response and the modelling scale."""

    @abstractmethod
    def fit(self, columns: Mapping[str, np.ndarray], y: np.ndarray) -> ResponseTransform: ...

    @abstractmethod
    def forward(self, columns: Mapping[str, np.ndarray], y: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def inverse(self, columns: Mapping[str, np.ndarray], z: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def describe(self) -> dict[str, Any]: ...

    def set_residual_correction(
        self, columns: Mapping[str, np.ndarray], y: np.ndarray, z_pred: np.ndarray
    ) -> None:
        """Optional hook to calibrate a back-transform bias correction."""
        return None


@dataclass
class IdentityResponse(ResponseTransform):
    """No transform; the model works on the physical scale."""

    def fit(self, columns: Mapping[str, np.ndarray], y: np.ndarray) -> IdentityResponse:
        return self

    def forward(self, columns: Mapping[str, np.ndarray], y: np.ndarray) -> np.ndarray:
        return np.asarray(y, dtype=np.float64)

    def inverse(self, columns: Mapping[str, np.ndarray], z: np.ndarray) -> np.ndarray:
        return np.asarray(z, dtype=np.float64)

    def describe(self) -> dict[str, Any]:
        return {"kind": "identity"}


@dataclass
class LogResponse(ResponseTransform):
    """Natural log of the response, with Duan smearing on the way back.

    ``smearing`` is the mean of ``exp(residual)`` over the training set.  It
    converts the back-transformed prediction from a conditional median-like
    quantity to an estimate of the conditional mean, which is what an energy or
    cost figure is expected to be.  It stays at 1.0 until
    :meth:`set_residual_correction` is called with in-sample predictions.
    """

    apply_smearing: bool = True
    smearing: float = 1.0

    def fit(self, columns: Mapping[str, np.ndarray], y: np.ndarray) -> LogResponse:
        y = np.asarray(y, dtype=np.float64)
        if np.any(y <= 0):
            raise ValueError("LogResponse requires strictly positive response values.")
        return self

    def forward(self, columns: Mapping[str, np.ndarray], y: np.ndarray) -> np.ndarray:
        return np.log(np.asarray(y, dtype=np.float64))

    def inverse(self, columns: Mapping[str, np.ndarray], z: np.ndarray) -> np.ndarray:
        return self.smearing * np.exp(np.asarray(z, dtype=np.float64))

    def set_residual_correction(
        self, columns: Mapping[str, np.ndarray], y: np.ndarray, z_pred: np.ndarray
    ) -> None:
        if not self.apply_smearing:
            return
        residuals = self.forward(columns, y) - np.asarray(z_pred, dtype=np.float64)
        finite = residuals[np.isfinite(residuals)]
        self.smearing = float(np.mean(np.exp(finite))) if finite.size else 1.0
        logger.debug("Duan smearing factor: %.6f", self.smearing)

    def describe(self) -> dict[str, Any]:
        return {"kind": "log", "smearing": self.smearing}


@dataclass
class DimensionlessLogResponse(ResponseTransform):
    """Divide the response by its Pi scale, then take logs.

    The regressor works on ``log(y / scale(x))``: the log departure of the
    response from the dimensionally consistent power law found by
    :mod:`modeling.features.dimensional`.  Both steps are exactly invertible,
    and Duan smearing is applied on the log step as in :class:`LogResponse`.
    """

    pi_basis: PiBasis
    apply_smearing: bool = True
    smearing: float = 1.0

    def fit(
        self, columns: Mapping[str, np.ndarray], y: np.ndarray
    ) -> DimensionlessLogResponse:
        group = self.pi_basis.to_response_group(columns, np.asarray(y, dtype=np.float64))
        if np.any(group <= 0) or not np.all(np.isfinite(group)):
            raise ValueError(
                "The Pi response group must be strictly positive and finite to be logged."
            )
        return self

    def forward(self, columns: Mapping[str, np.ndarray], y: np.ndarray) -> np.ndarray:
        group = self.pi_basis.to_response_group(columns, np.asarray(y, dtype=np.float64))
        return np.log(group)

    def inverse(self, columns: Mapping[str, np.ndarray], z: np.ndarray) -> np.ndarray:
        group = self.smearing * np.exp(np.asarray(z, dtype=np.float64))
        return self.pi_basis.from_response_group(columns, group)

    def set_residual_correction(
        self, columns: Mapping[str, np.ndarray], y: np.ndarray, z_pred: np.ndarray
    ) -> None:
        if not self.apply_smearing:
            return
        residuals = self.forward(columns, y) - np.asarray(z_pred, dtype=np.float64)
        finite = residuals[np.isfinite(residuals)]
        self.smearing = float(np.mean(np.exp(finite))) if finite.size else 1.0
        logger.debug("Duan smearing factor: %.6f", self.smearing)

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "dimensionless_log",
            "smearing": self.smearing,
            "response_group": self.pi_basis.response_group.formula(),
            "response_scale": self.pi_basis.scale_formula(),
        }


@dataclass
class FeatureSpace:
    """Builds the design matrix the regressor sees.

    ``raw``
        The original feature columns.
    ``pi``
        The dimensionless predictor groups.
    ``log_pi``
        Logs of the predictor groups, which turns a power law into a linear
        relation and is the easiest space for a symbolic search to work in.
    ``log_pi_plus_raw``
        The log predictor groups together with the log raw features.

    A Pi group formed from a variable that is already dimensionless may take
    non-positive values, and a group like that cannot be logged.  Such groups are
    passed through on their own scale and the fact is recorded in
    :attr:`logged`, so a formula read off the model stays correct.

    ``log_pi_plus_raw`` exists because strict dimensional similarity is an
    assumption, not a guarantee.  It holds only when every dimensional quantity
    that governs the response appears in the parameter table.  A simulator study
    that varies rock and well parameters while holding pipe geometry, fluid
    properties and the simulation horizon fixed violates it: those fixed
    quantities carry dimensions and are simply absent.  Reducing to the strict Pi
    basis then discards real degrees of freedom.  Keeping the raw features
    alongside the groups preserves the valuable half of the reduction -- the
    response is still measured against its dimensionally consistent power law,
    which is unit invariant and removes most of the spread -- without forcing an
    assumption the data does not support.
    """

    kind: str = "pi"
    pi_basis: PiBasis | None = None
    raw_names: tuple[str, ...] = ()
    names: tuple[str, ...] = field(default_factory=tuple)
    logged: tuple[bool, ...] = field(default_factory=tuple)

    KINDS = ("raw", "pi", "log_pi", "log_pi_plus_raw")

    def __post_init__(self) -> None:
        if self.kind not in self.KINDS:
            raise ValueError(
                f"Unknown FeatureSpace kind {self.kind!r}. Known: {list(self.KINDS)}"
            )
        if self.kind != "raw" and self.pi_basis is None:
            raise ValueError(f"FeatureSpace kind {self.kind!r} needs a pi_basis.")
        if self.kind == "raw":
            self.names = self.raw_names
        elif self.kind == "pi":
            self.names = self.pi_basis.predictor_names()  # type: ignore[union-attr]
        else:
            # Names are provisional until fit_columns sees which columns are positive.
            self.names = self._provisional_names()

    def _pi_names(self) -> tuple[str, ...]:
        return self.pi_basis.predictor_names()  # type: ignore[union-attr]

    def _source_names(self) -> tuple[str, ...]:
        """Columns feeding the space: groups first, then raw features."""
        if self.kind == "log_pi_plus_raw":
            return self._pi_names() + tuple(self.raw_names)
        return self._pi_names()

    def _provisional_names(self) -> tuple[str, ...]:
        return tuple(f"log_{name}" for name in self._source_names())

    def _raw_matrix(self, columns: Mapping[str, np.ndarray]) -> np.ndarray:
        return np.column_stack(
            [np.asarray(columns[name], dtype=np.float64) for name in self.raw_names]
        )

    def _source_matrix(self, columns: Mapping[str, np.ndarray]) -> np.ndarray:
        groups = self.pi_basis.transform(columns)  # type: ignore[union-attr]
        if self.kind == "log_pi_plus_raw" and self.raw_names:
            return np.column_stack([groups, self._raw_matrix(columns)])
        return groups

    def fit_columns(self, columns: Mapping[str, np.ndarray]) -> FeatureSpace:
        """Decide which columns can be logged, using the training design."""
        if self.kind not in {"log_pi", "log_pi_plus_raw"}:
            return self
        matrix = self._source_matrix(columns)
        self.logged = tuple(
            bool(np.all(matrix[:, index] > 0)) for index in range(matrix.shape[1])
        )
        source_names = self._source_names()
        self.names = tuple(
            f"log_{name}" if is_logged else name
            for name, is_logged in zip(source_names, self.logged)
        )
        skipped = [name for name, is_logged in zip(source_names, self.logged) if not is_logged]
        if skipped:
            logger.info("Columns kept on a linear scale (not strictly positive): %s", skipped)
        return self

    def transform(self, columns: Mapping[str, np.ndarray]) -> np.ndarray:
        if self.kind == "raw":
            return self._raw_matrix(columns)
        if self.kind == "pi":
            return self.pi_basis.transform(columns)  # type: ignore[union-attr]
        if not self.logged:
            self.fit_columns(columns)
        output = self._source_matrix(columns).astype(np.float64, copy=True)
        for index, is_logged in enumerate(self.logged):
            if is_logged:
                # Guard against a test point falling outside the training support.
                output[:, index] = np.log(
                    np.clip(output[:, index], np.finfo(np.float64).tiny, None)
                )
        return output

    def describe(self) -> dict[str, Any]:
        description: dict[str, Any] = {"kind": self.kind, "names": list(self.names)}
        if self.logged:
            description["logged"] = list(self.logged)
        if self.pi_basis is not None:
            description["groups"] = [
                group.to_dict() for group in self.pi_basis.predictor_groups
            ]
        return description


def columns_from_matrix(
    features: np.ndarray, feature_names: Sequence[str]
) -> dict[str, np.ndarray]:
    """Split a design matrix into the named-column mapping the transforms expect."""
    features = np.asarray(features, dtype=np.float64)
    return {name: features[:, index] for index, name in enumerate(feature_names)}
