import inspect

import numpy as np
import numpy.typing as npt
import sklearn.metrics as skm


_METRICS = {
    "d2_absolute_error_score",
    "d2_pinball_score",
    "d2_tweedie_score",
    "explained_variance_score",
    "max_error",
    "mean_absolute_error",
    "mean_absolute_percentage_error",
    "mean_gamma_deviance",
    "mean_pinball_loss",
    "mean_poisson_deviance",
    "mean_squared_error",
    "mean_squared_log_error",
    "mean_tweedie_deviance",
    "median_absolute_error",
    "r2_score",
    "root_mean_squared_error",
    "root_mean_squared_log_error",
}


def _compute_metrics_1d(
    y_actual: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
    sample_weight: np.ndarray | None,
) -> dict[str, float]:
    """Compute all regression metrics for a single pair of 1-D arrays."""
    valid = mask & np.isfinite(y_actual) & np.isfinite(y_pred)
    if valid.size == 0 or not np.any(valid):
        return {name: float("nan") for name in _METRICS}

    y_true_v = y_actual[valid]
    y_pred_v = y_pred[valid]
    sw = np.asarray(sample_weight)[valid] if sample_weight is not None else None

    results: dict[str, float] = {}
    for name, func in inspect.getmembers(skm, inspect.isfunction):
        if name not in _METRICS:
            continue
        try:
            sig = inspect.signature(func)
            kwargs: dict = {}
            if "sample_weight" in sig.parameters and sw is not None:
                kwargs["sample_weight"] = sw
            value = func(y_true_v, y_pred_v, **kwargs)
            results[name] = float(np.mean(value))
        except Exception:
            results[name] = float("nan")
    return results


def _prepare_arrays(
    y_actual: npt.NDArray[np.float64],
    y_pred: npt.NDArray[np.float64],
    mask: npt.NDArray[np.bool_] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalise inputs to 2-D ``(n_samples, n_targets)`` and build the row mask."""
    y_actual = np.asarray(y_actual, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_actual.ndim == 1:
        y_actual = y_actual[:, np.newaxis]
    if y_pred.ndim == 1:
        y_pred = y_pred[:, np.newaxis]
    n_samples = y_actual.shape[0]
    mask_1d = (
        np.ones(n_samples, dtype=bool)
        if mask is None
        else np.asarray(mask, dtype=bool).flatten()
    )
    return y_actual, y_pred, mask_1d


def run_regression_metrics_per_target(
    y_actual: npt.NDArray[np.float64],
    y_pred: npt.NDArray[np.float64],
    target_names: tuple[str, ...] | None = None,
    mask: npt.NDArray[np.bool_] | None = None,
    *,
    sample_weight: npt.NDArray[np.float64] | None = None,
) -> dict[str, dict[str, float]]:
    """Compute all sklearn regression metrics independently for each target column.

    Returns a dict keyed by target name whose values are per-metric dicts::

        {
            "pressure":    {"mean_squared_error": 0.12, "r2_score": 0.91, ...},
            "temperature": {"mean_squared_error": 0.07, "r2_score": 0.95, ...},
        }

    For single-target inputs the dict has exactly one key (``target_names[0]``
    or ``"target_0"`` by default).
    """
    y_actual, y_pred, mask_1d = _prepare_arrays(y_actual, y_pred, mask)
    n_targets = y_actual.shape[1]

    if target_names is None:
        target_names = tuple(f"target_{k}" for k in range(n_targets))
    elif len(target_names) != n_targets:
        raise ValueError(
            f"len(target_names)={len(target_names)} does not match "
            f"n_targets={n_targets}"
        )

    return {
        target_names[k]: _compute_metrics_1d(
            y_actual[:, k], y_pred[:, k], mask_1d, sample_weight
        )
        for k in range(n_targets)
    }


def run_all_regression_metrics(
    y_actual: npt.NDArray[np.float64],
    y_pred: npt.NDArray[np.float64],
    mask: npt.NDArray[np.bool_] | None = None,
    *,
    sample_weight: npt.NDArray[np.float64] | None = None,
) -> dict[str, float]:
    """Compute all sklearn regression metrics, averaged across targets.

    For multi-target inputs (shape ``(n_samples, n_targets)``), metrics are
    computed independently per target column and then averaged across targets.

    For single-target inputs (shape ``(n_samples,)`` or ``(n_samples, 1)``),
    behaviour is identical to before.

    Returns NaN for a metric when no valid samples remain for *all* targets.
    """
    y_actual, y_pred, mask_1d = _prepare_arrays(y_actual, y_pred, mask)
    n_targets = y_actual.shape[1]

    per_target = [
        _compute_metrics_1d(y_actual[:, k], y_pred[:, k], mask_1d, sample_weight)
        for k in range(n_targets)
    ]

    if not per_target:
        return {}

    all_keys = per_target[0].keys()
    return {
        key: (
            float(np.mean([d[key] for d in per_target if not np.isnan(d[key])]))
            if any(not np.isnan(d[key]) for d in per_target)
            else float("nan")
        )
        for key in all_keys
    }