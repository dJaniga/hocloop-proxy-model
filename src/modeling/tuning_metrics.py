import sklearn.metrics as skm

MAXIMIZE_METRICS = {
    "d2_absolute_error_score",
    "d2_pinball_score",
    "d2_tweedie_score",
    "explained_variance_score",
    "r2_score",
}

AVAILABLE_METRICS = [
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
]


def get_metric_direction(metric_name: str) -> str:
    """Return 'maximize' or 'minimize' depending on the metric."""
    if metric_name in MAXIMIZE_METRICS:
        return "maximize"
    return "minimize"


def evaluate_metric(metric_name: str, y_true, y_pred) -> float:
    """Evaluate a specific sklearn regression metric."""

    if not hasattr(skm, metric_name):
        raise ValueError(f"Metric {metric_name} not found in sklearn.metrics")

    func = getattr(skm, metric_name)
    try:
        return float(func(y_true, y_pred))
    except Exception:
        # Return a terrible score if metric fails (e.g. log metrics with negative values)
        return float("-inf") if metric_name in MAXIMIZE_METRICS else float("inf")