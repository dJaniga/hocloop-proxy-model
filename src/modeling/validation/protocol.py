"""Validation protocols for a simulator proxy.

Plain k-fold answers one question: how well does the proxy interpolate inside
the region the design already covers?  A proxy is normally built to be used
somewhere else, so this module adds two protocols that test what a reviewer will
ask about.

**Extrapolation shell.**  A Latin hypercube fills a box.  Training on the
interior and testing on the outer shell measures what happens when the proxy is
queried beyond the region it learnt from, which is the failure mode that matters
in practice and is invisible to random splits.

**Sample efficiency.**  Every training point costs one simulator run, so the
useful quantity is not accuracy at the full design size but the design size
needed to reach a target accuracy.  The learning curve turns the proxy's value
into a number of simulator runs saved.

All splitting is stratified on binned log response, so folds stay comparable
across a target that spans several orders of magnitude.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Sequence

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from modeling.fit_metrics import run_regression_metrics_per_target

logger = logging.getLogger(__name__)

ModelFactory = Callable[[], Any]


def stratification_bins(y: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Quantile bins of the response, used to keep folds comparable.

    A response spanning orders of magnitude is binned on a log scale so that the
    bins are quantiles of the actual distribution rather than of its tail.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    values = np.log(y) if np.all(y > 0) else y
    quantiles = np.quantile(values, np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.digitize(values, quantiles)


@dataclass
class FoldResult:
    """Per-target metrics from one evaluation fold."""

    fold: int
    n_train: int
    n_test: int
    metrics: dict[str, dict[str, float]]
    extra: dict[str, Any] = field(default_factory=dict)


def _fit_predict(
    model_factory: ModelFactory,
    features: np.ndarray,
    targets: np.ndarray,
    features_name: Sequence[str],
    targets_name: Sequence[str],
    train_index: np.ndarray,
    test_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, Any]:
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
    return targets[test_index], predicted, model


def repeated_stratified_cv(
    model_factory: ModelFactory,
    features: np.ndarray,
    targets: np.ndarray,
    features_name: Sequence[str],
    targets_name: Sequence[str],
    n_splits: int = 5,
    n_repeats: int = 2,
    stratify_on: int = 0,
    seed: int | None = 0,
) -> list[FoldResult]:
    """Repeated stratified k-fold over the whole design.

    Repeats reduce the dependence of the reported score on one particular
    partition, which matters when a single fold's estimate is compared against a
    competing method.
    """
    features = np.asarray(features, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if targets.ndim == 1:
        targets = targets.reshape(-1, 1)

    bins = stratification_bins(targets[:, stratify_on])
    splitter = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=seed
    )

    results: list[FoldResult] = []
    for fold, (train_index, test_index) in enumerate(
        splitter.split(features, bins), start=1
    ):
        actual, predicted, _ = _fit_predict(
            model_factory, features, targets, features_name, targets_name,
            train_index, test_index,
        )
        results.append(
            FoldResult(
                fold=fold,
                n_train=int(train_index.size),
                n_test=int(test_index.size),
                metrics=run_regression_metrics_per_target(
                    actual, predicted, target_names=tuple(targets_name)
                ),
            )
        )
        logger.info("CV fold %d/%d complete.", fold, n_splits * n_repeats)
    return results


def shell_split(
    features: np.ndarray,
    shell_fraction: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Split a box-filling design into an interior core and an outer shell.

    Each feature is mapped to its position within its own observed range, and a
    sample belongs to the shell when any coordinate sits in the outer margin of
    that range at either end.  Training on the core and testing on the shell
    forces every test point to lie outside the training envelope in at least one
    dimension.

    ``shell_fraction`` is the fraction of *samples* to place in the shell, not
    the margin applied per feature.  The distinction matters: a margin trims each
    dimension independently, so a fixed margin removes a fraction of the samples
    that grows with dimension.  Trimming a quarter of every axis of a
    seven-dimensional design leaves only ``0.75 ** 7``, about 13%, in the core --
    which would measure small-sample behaviour and extrapolation at the same
    time and let neither be read off the result.  The per-feature margin is
    therefore solved from the requested sample fraction, holding the split
    interpretable as the dimension changes.
    """
    features = np.asarray(features, dtype=np.float64)
    n_samples, n_features = features.shape
    if not 0.0 < shell_fraction < 1.0:
        raise ValueError(f"shell_fraction must lie in (0, 1); got {shell_fraction}.")

    minimum = features.min(axis=0)
    maximum = features.max(axis=0)
    span = np.where(maximum > minimum, maximum - minimum, 1.0)
    position = (features - minimum) / span

    # Retaining a fraction q of each of d independent axes leaves q ** d of a
    # uniform design in the core, so q = (1 - shell_fraction) ** (1 / d).
    keep_per_axis = (1.0 - shell_fraction) ** (1.0 / n_features)
    margin = (1.0 - keep_per_axis) / 2.0

    in_shell = np.any((position < margin) | (position > 1.0 - margin), axis=1)
    core_index = np.flatnonzero(~in_shell)
    shell_index = np.flatnonzero(in_shell)
    logger.info(
        "Shell split: %d core samples, %d shell samples (%.1f%% held out; "
        "per-feature margin %.3f of the range at each end).",
        core_index.size,
        shell_index.size,
        100.0 * shell_index.size / n_samples,
        margin,
    )
    return core_index, shell_index


def extrapolation_test(
    model_factory: ModelFactory,
    features: np.ndarray,
    targets: np.ndarray,
    features_name: Sequence[str],
    targets_name: Sequence[str],
    shell_fraction: float = 0.25,
) -> FoldResult:
    """Train on the design interior, test on the outer shell."""
    targets = np.asarray(targets, dtype=np.float64)
    if targets.ndim == 1:
        targets = targets.reshape(-1, 1)
    core_index, shell_index = shell_split(features, shell_fraction)
    if core_index.size == 0 or shell_index.size == 0:
        raise ValueError(
            f"shell_fraction={shell_fraction} leaves one side of the split empty."
        )
    actual, predicted, _ = _fit_predict(
        model_factory, np.asarray(features, dtype=np.float64), targets,
        features_name, targets_name, core_index, shell_index,
    )
    return FoldResult(
        fold=0,
        n_train=int(core_index.size),
        n_test=int(shell_index.size),
        metrics=run_regression_metrics_per_target(
            actual, predicted, target_names=tuple(targets_name)
        ),
        extra={"shell_fraction": shell_fraction},
    )


def learning_curve(
    model_factory: ModelFactory,
    features: np.ndarray,
    targets: np.ndarray,
    features_name: Sequence[str],
    targets_name: Sequence[str],
    train_sizes: Sequence[int] = (100, 200, 400, 800, 1600, 3200),
    n_repeats: int = 3,
    holdout_fraction: float = 0.2,
    stratify_on: int = 0,
    seed: int | None = 0,
) -> list[FoldResult]:
    """Accuracy as a function of the number of simulator runs used for training.

    A fixed hold-out set is drawn once per repeat and every training size is
    evaluated against it, so points on the curve differ only in training size.
    Training subsets are stratified so a small subset still spans the response
    range instead of concentrating where the design is dense.
    """
    features = np.asarray(features, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if targets.ndim == 1:
        targets = targets.reshape(-1, 1)

    bins = stratification_bins(targets[:, stratify_on])
    rng = np.random.default_rng(seed)
    n_holdout_splits = max(2, int(round(1.0 / holdout_fraction)))

    results: list[FoldResult] = []
    for repeat in range(n_repeats):
        splitter = StratifiedKFold(
            n_splits=n_holdout_splits, shuffle=True, random_state=int(rng.integers(2**31))
        )
        pool_index, holdout_index = next(iter(splitter.split(features, bins)))

        pool_bins = bins[pool_index]
        for size in train_sizes:
            if size >= pool_index.size:
                logger.info(
                    "Skipping train size %d: only %d samples available.",
                    size,
                    pool_index.size,
                )
                continue
            # Stratified subsample of the pool, so small designs still span the
            # response range instead of clustering where the design is dense.
            fraction = size / pool_index.size
            subset: list[int] = []
            for value in np.unique(pool_bins):
                candidates = pool_index[pool_bins == value]
                take = max(1, int(round(fraction * candidates.size)))
                subset.extend(
                    rng.choice(candidates, size=min(take, candidates.size), replace=False)
                )
            # Rounding per bin overshoots the requested size, so the list is
            # shuffled before it is trimmed. Trimming the tail directly would
            # drop whole response bins and undo the stratification.
            subset_array = np.array(subset, dtype=int)
            rng.shuffle(subset_array)
            train_index = subset_array[:size]

            actual, predicted, _ = _fit_predict(
                model_factory, features, targets, features_name, targets_name,
                train_index, holdout_index,
            )
            results.append(
                FoldResult(
                    fold=repeat,
                    n_train=int(train_index.size),
                    n_test=int(holdout_index.size),
                    metrics=run_regression_metrics_per_target(
                        actual, predicted, target_names=tuple(targets_name)
                    ),
                    extra={"requested_train_size": int(size)},
                )
            )
            logger.info("Learning curve: repeat %d, train size %d.", repeat, size)
    return results


def aggregate(results: Sequence[FoldResult], metric: str = "r2_score") -> dict[str, dict[str, float]]:
    """Mean and standard deviation of one metric per target across folds."""
    if not results:
        return {}
    summary: dict[str, dict[str, float]] = {}
    for target in results[0].metrics:
        values = np.array(
            [result.metrics[target][metric] for result in results], dtype=np.float64
        )
        finite = values[np.isfinite(values)]
        summary[target] = {
            "mean": float(np.mean(finite)) if finite.size else float("nan"),
            "std": float(np.std(finite)) if finite.size else float("nan"),
            "min": float(np.min(finite)) if finite.size else float("nan"),
            "n_folds": int(values.size),
        }
    return summary
