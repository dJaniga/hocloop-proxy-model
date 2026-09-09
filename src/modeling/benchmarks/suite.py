"""Benchmark and ablation suite.

Two tables come out of this module.

The **learner comparison** holds the pipeline fixed and swaps only the stage-3
regressor, so the symbolic models and the black-box baselines are measured
against each other on identical inputs.  Differences are attributable to the
regressor rather than to feature engineering done for one contender and not
another.

The **ablation** holds the learner fixed and switches pipeline stages off one at
a time, which is what shows whether structure discovery and dimensional
reduction earn their place or merely accompany a good regressor.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Sequence

import numpy as np

from modeling.proxy import ProxyConfig, StructuredProxyModel
from modeling.validation.protocol import (
    FoldResult,
    aggregate,
    extrapolation_test,
    repeated_stratified_cv,
)

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkEntry:
    """Results for one configuration."""

    label: str
    cv: dict[str, dict[str, float]]
    extrapolation: dict[str, dict[str, float]] = field(default_factory=dict)
    closed_form: dict[str, str] = field(default_factory=dict)
    failed: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "cv": self.cv,
            "extrapolation": self.extrapolation,
            "closed_form": self.closed_form,
            "failed": self.failed,
        }


def _metrics_by_target(results: Sequence[FoldResult], metrics: Sequence[str]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for metric in metrics:
        for target, values in aggregate(results, metric).items():
            summary.setdefault(target, {})[f"{metric}_mean"] = values["mean"]
            summary.setdefault(target, {})[f"{metric}_std"] = values["std"]
    return summary


def evaluate_configuration(
    label: str,
    model_factory: Callable[[], StructuredProxyModel],
    features: np.ndarray,
    targets: np.ndarray,
    features_name: Sequence[str],
    targets_name: Sequence[str],
    n_splits: int = 5,
    n_repeats: int = 1,
    metrics: Sequence[str] = (
        "r2_score",
        "r2_log_score",
        "mean_absolute_percentage_error",
        "root_mean_squared_error",
    ),
    run_extrapolation: bool = True,
    shell_fraction: float = 0.25,
    seed: int | None = 0,
) -> BenchmarkEntry:
    """Cross-validate one configuration and, optionally, stress it on the shell."""
    logger.info("=== Benchmark: %s ===", label)
    try:
        cv_results = repeated_stratified_cv(
            model_factory, features, targets, features_name, targets_name,
            n_splits=n_splits, n_repeats=n_repeats, seed=seed,
        )
        cv_summary = _metrics_by_target(cv_results, metrics)

        extrapolation_summary: dict[str, dict[str, float]] = {}
        if run_extrapolation:
            shell = extrapolation_test(
                model_factory, features, targets, features_name, targets_name,
                shell_fraction=shell_fraction,
            )
            extrapolation_summary = {
                target: {metric: values[metric] for metric in metrics if metric in values}
                for target, values in shell.metrics.items()
            }

        # One model fitted on everything, purely to report the formula it found.
        final = model_factory()
        final.fit(features, targets, tuple(features_name), tuple(targets_name))
        closed_form = final.closed_form() if hasattr(final, "closed_form") else {}

        return BenchmarkEntry(
            label=label,
            cv=cv_summary,
            extrapolation=extrapolation_summary,
            closed_form=closed_form,
        )
    except Exception as error:  # A single contender must not abort the suite.
        logger.exception("Benchmark %r failed.", label)
        return BenchmarkEntry(label=label, cv={}, failed=f"{type(error).__name__}: {error}")


def learner_comparison(
    learner_factories: dict[str, Callable[[], Any]],
    config: ProxyConfig,
    features: np.ndarray,
    targets: np.ndarray,
    features_name: Sequence[str],
    targets_name: Sequence[str],
    model_factory_for: Callable[[str, Callable[[], Any]], Callable[[], Any]] | None = None,
    **kwargs: Any,
) -> list[BenchmarkEntry]:
    """Swap the stage-3 learner, holding every other stage fixed.

    ``model_factory_for`` maps a learner label and its factory to the model
    factory actually evaluated.  It is the hook the hyperparameter search uses:
    returning a self-tuning model there makes every entry in the table a nested
    cross-validation, because the search only ever sees an outer training split.
    """
    entries: list[BenchmarkEntry] = []
    for label, learner_factory in learner_factories.items():
        if model_factory_for is not None:
            model_factory = model_factory_for(label, learner_factory)
        else:
            def model_factory(_factory=learner_factory) -> StructuredProxyModel:
                return StructuredProxyModel(learner_factory=_factory, config=config)

        entries.append(
            evaluate_configuration(
                label, model_factory, features, targets, features_name, targets_name, **kwargs
            )
        )
    return entries


#: Pipeline variants used by the ablation, as overrides applied to the full config.
ABLATIONS: dict[str, dict[str, Any]] = {
    "full_pipeline": {},
    "no_structure_discovery": {"use_structure_discovery": False},
    "no_dimensional_reduction": {
        "use_dimensional_reduction": False,
        "feature_space": "raw",
    },
    "no_power_law_refinement": {"refine_power_law": False},
    "no_log_response": {"log_response": False},
    "raw_features_only": {
        "use_structure_discovery": False,
        "use_dimensional_reduction": False,
        "feature_space": "raw",
        "log_response": False,
    },
}


def ablation_study(
    learner_factory: Callable[[], Any],
    config: ProxyConfig,
    features: np.ndarray,
    targets: np.ndarray,
    features_name: Sequence[str],
    targets_name: Sequence[str],
    variants: dict[str, dict[str, Any]] | None = None,
    **kwargs: Any,
) -> list[BenchmarkEntry]:
    """Switch pipeline stages off one at a time, holding the learner fixed."""
    variants = variants if variants is not None else ABLATIONS
    entries: list[BenchmarkEntry] = []
    for label, overrides in variants.items():
        variant_config = replace(config, **overrides)

        def model_factory(_config=variant_config) -> StructuredProxyModel:
            return StructuredProxyModel(learner_factory=learner_factory, config=_config)

        entries.append(
            evaluate_configuration(
                label, model_factory, features, targets, features_name, targets_name, **kwargs
            )
        )
    return entries


def to_dataframe(entries: Sequence[BenchmarkEntry], metric: str = "r2_score_mean") -> Any:
    """Flatten benchmark entries into a table with one row per configuration."""
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for entry in entries:
        row: dict[str, Any] = {"configuration": entry.label}
        if entry.failed:
            row["failed"] = entry.failed
        for target, values in entry.cv.items():
            row[f"cv_{target}"] = values.get(metric)
        for target, values in entry.extrapolation.items():
            row[f"shell_{target}"] = values.get(metric.replace("_mean", ""))
        rows.append(row)
    return pd.DataFrame(rows)
