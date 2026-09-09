"""Distribution-free prediction intervals for the proxy model.

A proxy that reports only a point estimate cannot be used for a decision that
depends on how wrong it might be.  Split and CV+ conformal prediction supply
finite-sample coverage guarantees that hold for any underlying model and make no
assumption about the error distribution, which suits a surrogate whose residuals
are neither Gaussian nor homoscedastic.

Two properties of this pipeline make conformal prediction fit particularly well.

*Errors are multiplicative.*  The model is fitted on a log scale, so conformity
is scored there too.  A constant interval in log space is a constant *relative*
interval in physical units, which matches how surrogate error actually behaves
across four orders of magnitude of output.

*Intervals propagate exactly.*  Derived targets are reconstructed through exact
identities, so an interval on an independent target maps to its derived targets
by evaluating the identity at the interval endpoints.  For a relation such as
``LCOH = C(L) / w_out`` the map is exact and adds no width of its own: the
relative uncertainty of the derived target equals that of the independent one.
No error budget has to be assumed or simulated.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.model_selection import KFold

from modeling.structure.identities import TargetStructure

logger = logging.getLogger(__name__)


@dataclass
class PredictionInterval:
    """Lower and upper bounds with the nominal coverage they were built for."""

    lower: np.ndarray
    upper: np.ndarray
    alpha: float

    @property
    def nominal_coverage(self) -> float:
        return 1.0 - self.alpha

    def contains(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64)
        return (y >= self.lower) & (y <= self.upper)

    def empirical_coverage(self, y: np.ndarray) -> float:
        return float(np.mean(self.contains(y)))

    def mean_width(self) -> float:
        return float(np.mean(self.upper - self.lower))

    def mean_relative_width(self, y_pred: np.ndarray) -> float:
        y_pred = np.asarray(y_pred, dtype=np.float64)
        safe = np.where(np.abs(y_pred) > 0, np.abs(y_pred), 1.0)
        return float(np.mean((self.upper - self.lower) / safe))


@dataclass
class LogScaleConformalCalibrator:
    """Calibrates a symmetric conformal radius on the log scale.

    The conformity score of a sample is ``|log y - log yhat|``.  The radius is
    the ``ceil((n + 1) * (1 - alpha)) / n`` empirical quantile of the
    out-of-fold scores, the finite-sample correction that gives CV+ its
    guarantee rather than the plain empirical quantile.

    Residuals are collected out of fold, so the calibration never sees a
    prediction the model made on data it was trained on.
    """

    alpha: float = 0.1
    n_splits: int = 10
    seed: int | None = 0
    radius_: float = float("nan")
    scores_: np.ndarray = field(default_factory=lambda: np.empty(0))

    def calibrate_from_scores(self, scores: np.ndarray) -> LogScaleConformalCalibrator:
        scores = np.asarray(scores, dtype=np.float64)
        scores = scores[np.isfinite(scores)]
        if scores.size == 0:
            raise ValueError("No finite conformity scores available for calibration.")
        n = scores.size
        level = min(1.0, np.ceil((n + 1) * (1.0 - self.alpha)) / n)
        self.scores_ = scores
        self.radius_ = float(np.quantile(scores, level, method="higher"))
        logger.info(
            "Conformal radius at %.0f%% nominal coverage: %.5f log units "
            "(multiplicative factor %.4f)",
            100 * (1 - self.alpha),
            self.radius_,
            float(np.exp(self.radius_)),
        )
        return self

    def interval(self, y_pred: np.ndarray) -> PredictionInterval:
        """Multiplicative interval around a positive point prediction."""
        if not np.isfinite(self.radius_):
            raise ValueError("Calibrator has not been calibrated yet.")
        y_pred = np.asarray(y_pred, dtype=np.float64)
        factor = float(np.exp(self.radius_))
        return PredictionInterval(
            lower=y_pred / factor, upper=y_pred * factor, alpha=self.alpha
        )


def cross_validated_log_scores(
    model_factory,
    features: np.ndarray,
    targets: np.ndarray,
    features_name: Sequence[str],
    targets_name: Sequence[str],
    target: str,
    n_splits: int = 10,
    seed: int | None = 0,
) -> np.ndarray:
    """Out-of-fold ``|log y - log yhat|`` scores for one target.

    ``model_factory`` must return a fresh, unfitted model each call so that no
    fold is scored by a model that saw it.
    """
    features = np.asarray(features, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if targets.ndim == 1:
        targets = targets.reshape(-1, 1)
    target_index = list(targets_name).index(target)

    scores = np.full(features.shape[0], np.nan, dtype=np.float64)
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (train_index, test_index) in enumerate(folds.split(features), start=1):
        model = model_factory()
        model.fit(
            features[train_index],
            targets[train_index],
            tuple(features_name),
            tuple(targets_name),
        )
        predicted = np.asarray(model.predict(features[test_index]), dtype=np.float64)
        if predicted.ndim == 1:
            predicted = predicted.reshape(-1, 1)
        actual = targets[test_index, target_index]
        estimate = predicted[:, target_index]
        valid = (actual > 0) & (estimate > 0)
        block = np.full(test_index.size, np.nan)
        block[valid] = np.abs(np.log(actual[valid]) - np.log(estimate[valid]))
        scores[test_index] = block
        logger.debug("Conformal fold %d/%d scored.", fold, n_splits)
    return scores


def propagate_interval(
    structure: TargetStructure,
    independent_intervals: Mapping[str, PredictionInterval],
    point_predictions: Mapping[str, np.ndarray],
    feature_columns: Mapping[str, np.ndarray],
) -> dict[str, PredictionInterval]:
    """Map intervals from the independent targets onto every derived target.

    Each identity is evaluated at the interval endpoints and the results are
    sorted, which is exact whenever the identity is monotone in the source
    target.  Every family the discovery searches -- a positive multiple, an
    affine map, and ``P(x) / y`` on positive data -- is monotone, so the
    resulting interval inherits the coverage of the interval it came from
    without any additional assumption.
    """
    alpha = next(iter(independent_intervals.values())).alpha
    lower_values = {name: interval.lower for name, interval in independent_intervals.items()}
    upper_values = {name: interval.upper for name, interval in independent_intervals.items()}
    point_values = dict(point_predictions)

    intervals: dict[str, PredictionInterval] = dict(independent_intervals)
    for name in structure.reconstruction_order():
        identity = structure.identities[name]
        first = identity(lower_values, feature_columns)
        second = identity(upper_values, feature_columns)
        lower = np.minimum(first, second)
        upper = np.maximum(first, second)
        lower_values[name] = lower
        upper_values[name] = upper
        point_values[name] = identity(point_values, feature_columns)
        intervals[name] = PredictionInterval(lower=lower, upper=upper, alpha=alpha)
    return intervals


def coverage_report(
    intervals: Mapping[str, PredictionInterval],
    actuals: Mapping[str, np.ndarray],
    point_predictions: Mapping[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    """Empirical coverage and width per target, for the validation table."""
    report: dict[str, dict[str, float]] = {}
    for name, interval in intervals.items():
        if name not in actuals:
            continue
        report[name] = {
            "nominal_coverage": interval.nominal_coverage,
            "empirical_coverage": interval.empirical_coverage(actuals[name]),
            "mean_width": interval.mean_width(),
            "mean_relative_width": interval.mean_relative_width(
                point_predictions.get(name, interval.lower)
            ),
        }
    return report
