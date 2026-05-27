import copy
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import orjson
import pandas as pd
from sklearn.model_selection import KFold

from modeling.fit_metrics import run_all_regression_metrics, run_regression_metrics_per_target

logger = logging.getLogger(__name__)


class Regressor(ABC):

    @abstractmethod
    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        features_name: tuple[str, ...] | None = None,
        targets_name: tuple[str, ...] | None = None,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> Regressor: ...

    @abstractmethod
    def predict(self, features: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def __str__(self) -> str: ...

    @abstractmethod
    def get_fit_details(self) -> dict[str, Any]: ...


def _weighted_average_metrics(
    metrics_list: list[dict[str, float]],
    fold_sizes: list[int],
) -> dict[str, float]:
    """Weighted average of per-fold metric dicts, weighted by fold size.

    NaN values in individual folds are excluded from the average; if all folds
    are NaN for a metric the result is NaN.
    """
    if not metrics_list:
        return {}
    result: dict[str, float] = {}
    for key in metrics_list[0].keys():
        values_and_weights = [
            (m[key], fold_sizes[i])
            for i, m in enumerate(metrics_list)
            if m.get(key) is not None and not np.isnan(m[key])
        ]
        if values_and_weights:
            scores, weights = zip(*values_and_weights)
            result[key] = float(np.average(scores, weights=weights))
        else:
            result[key] = float("nan")
    return result


def _resolve_targets_name(
    targets_name: tuple[str, ...] | None,
    n_targets: int,
) -> tuple[str, ...]:
    if targets_name is not None and len(targets_name) == n_targets:
        return targets_name
    return tuple(f"target_{k}" for k in range(n_targets))


@dataclass
class ModelWrapper:
    model: Regressor
    output_path: Path
    seed: int | None = None
    n_jobs: int | None = None

    def __post_init__(self) -> None:
        if self.seed is None:
            self.seed = getattr(self.model, "seed", None)

    def fit(
        self,
        index: np.ndarray,
        features: np.ndarray,
        targets: np.ndarray,
        features_name: tuple[str, ...] | None = None,
        targets_name: tuple[str, ...] | None = None,
        optimize_hyperparameters: bool = False,
        tuning_metric: str = "mean_squared_error",
        outer_splits: int = 5,
        inner_splits: int = 5,
    ) -> Regressor:
        _features_name: tuple[str, ...] = (
            features_name
            if features_name is not None
            else tuple(f"ARG{i}" for i in range(features.shape[1]))
        )

        features = np.asarray(features)
        targets = np.asarray(targets)

        n_targets = targets.shape[1] if targets.ndim == 2 else 1
        _targets_name = _resolve_targets_name(targets_name, n_targets)

        # ------------------------------------------------------------------ #
        # 1. NESTED CROSS-VALIDATION — unbiased generalization estimate        #
        #    Outer fold: held-out eval                                         #
        #    Inner fold (optional): hyperparameter tuning                      #
        # ------------------------------------------------------------------ #
        outer_kf = KFold(n_splits=outer_splits, shuffle=True, random_state=self.seed)

        # Summary (averaged across targets) — one dict per fold
        outer_metrics_list: list[dict[str, float]] = []
        # Per-target — outer list is folds, inner dict is target_name -> metric_dict
        outer_per_target_list: list[dict[str, dict[str, float]]] = []
        outer_fold_sizes: list[int] = []

        for outer_idx, (outer_train_idx, outer_val_idx) in enumerate(
            outer_kf.split(features)
        ):
            logger.debug(f"Outer CV fold {outer_idx + 1}/{outer_splits}")

            X_outer_train, X_outer_val = (
                features[outer_train_idx],
                features[outer_val_idx],
            )
            y_outer_train, y_outer_val = (
                targets[outer_train_idx],
                targets[outer_val_idx],
            )

            fold_model = copy.deepcopy(self.model)

            if optimize_hyperparameters:
                from modeling.tuning import tune_hyperparameters

                fold_model = tune_hyperparameters(
                    fold_model,
                    X_outer_train,
                    y_outer_train,
                    tuning_metric=tuning_metric,
                    features_name=_features_name,
                    targets_name=_targets_name,
                    seed=self.seed,
                    n_splits=inner_splits,
                    n_jobs=self.n_jobs,
                )

            fold_model.fit(
                X_outer_train,
                y_outer_train,
                features_name=_features_name,
                targets_name=_targets_name,
                eval_set=(X_outer_val, y_outer_val),
            )

            y_outer_pred = fold_model.predict(X_outer_val)
            outer_metrics_list.append(
                run_all_regression_metrics(y_outer_val, y_outer_pred)
            )
            outer_per_target_list.append(
                run_regression_metrics_per_target(
                    y_outer_val, y_outer_pred, target_names=_targets_name
                )
            )
            outer_fold_sizes.append(len(outer_val_idx))

        # Weighted average across outer folds — summary
        nested_cv_metrics = _weighted_average_metrics(outer_metrics_list, outer_fold_sizes)

        # Weighted average across outer folds — per target
        # nested_cv_per_target: {target_name: {metric_name: float}}
        nested_cv_per_target: dict[str, dict[str, float]] = {}
        for tname in _targets_name:
            fold_metric_dicts = [fold[tname] for fold in outer_per_target_list]
            nested_cv_per_target[tname] = _weighted_average_metrics(
                fold_metric_dicts, outer_fold_sizes
            )

        logger.debug(
            "Nested CV complete",
            extra={
                "nested_cv_metrics": nested_cv_metrics,
                "nested_cv_per_target": nested_cv_per_target,
            },
        )

        # ------------------------------------------------------------------ #
        # 2. FINAL MODEL — trained on ALL data                                #
        #    Hyperparameters tuned on all data via inner CV (no holdout leak) #
        # ------------------------------------------------------------------ #
        logger.debug("Training final model on full dataset")

        if optimize_hyperparameters:
            from modeling.tuning import tune_hyperparameters

            self.model = tune_hyperparameters(
                self.model,
                features,
                targets,
                tuning_metric=tuning_metric,
                features_name=_features_name,
                targets_name=_targets_name,
                seed=self.seed,
                n_splits=inner_splits,
                n_jobs=self.n_jobs,
            )

        self.model.fit(
            features, targets,
            features_name=_features_name,
            targets_name=_targets_name,
        )

        # Training-set metrics (optimistic, diagnostic only)
        y_pred_train_all = self.predict(features)
        train_metrics = run_all_regression_metrics(targets, y_pred_train_all)
        train_per_target = run_regression_metrics_per_target(
            targets, y_pred_train_all, target_names=_targets_name
        )

        fit_metrics = {
            "train_resubstitution": train_metrics,
            "nested_cv": nested_cv_metrics,
        }
        # Per-target structure:
        # {target_name: {"train_resubstitution": {...}, "nested_cv": {...}}}
        fit_metrics_per_target: dict[str, dict[str, dict[str, float]]] = {
            tname: {
                "train_resubstitution": train_per_target[tname],
                "nested_cv": nested_cv_per_target[tname],
            }
            for tname in _targets_name
        }

        logger.debug("Fit diagnostics", extra={"fit_metrics": fit_metrics})
        logger.info(
            f"Model {self.model} fit completed with target metric "
            f"{tuning_metric}: {train_metrics.get(tuning_metric)}"
        )

        # ------------------------------------------------------------------ #
        # 3. EXPORT                                                            #
        # ------------------------------------------------------------------ #
        self.output_path.mkdir(parents=True, exist_ok=True)
        model_name = str(self.model)

        # --- summary metrics (averaged across targets) ----------------------
        with open(
            Path(self.output_path, f"{model_name}_fit_results").with_suffix(".json"), "w"
        ) as f:
            json.dump(fit_metrics, f, indent=4)

        pd.DataFrame(fit_metrics).reset_index().rename(
            columns={"index": "Metric"}
        ).to_csv(
            Path(self.output_path, f"{model_name}_fit_results").with_suffix(".csv"),
            index=False,
        )

        # --- per-target metrics ---------------------------------------------
        with open(
            Path(self.output_path, f"{model_name}_fit_results_per_target").with_suffix(".json"),
            "w",
        ) as f:
            json.dump(fit_metrics_per_target, f, indent=4)

        # Long-form CSV: columns are [Target, Split, Metric, Value]
        # Each row is one (target, split, metric) triple — easy to pivot/filter.
        per_target_rows = []
        for tname, splits in fit_metrics_per_target.items():
            for split_name, metric_dict in splits.items():
                for metric_name, value in metric_dict.items():
                    per_target_rows.append(
                        {
                            "Target": tname,
                            "Split": split_name,
                            "Metric": metric_name,
                            "Value": value,
                        }
                    )
        pd.DataFrame(per_target_rows).to_csv(
            Path(self.output_path, f"{model_name}_fit_results_per_target").with_suffix(".csv"),
            index=False,
        )

        # --- model internals ------------------------------------------------
        with open(
            Path(self.output_path, f"{model_name}_fit_details").with_suffix(".json"), "wb"
        ) as f:
            f.write(
                orjson.dumps(
                    self.model.get_fit_details(), option=orjson.OPT_SERIALIZE_NUMPY
                )
            )

        # --- predictions on full data ---------------------------------------
        y_pred_all = self.predict(features)
        _targets_arr = targets if targets.ndim == 2 else targets[:, np.newaxis]
        _preds_arr = y_pred_all if y_pred_all.ndim == 2 else y_pred_all[:, np.newaxis]

        df_content: dict = {"T": index.tolist()}
        df_content.update({k: v for k, v in zip(_features_name, features.T.tolist())})
        for k, col_name in enumerate(_targets_name):
            df_content[f"actual_{col_name}"] = _targets_arr[:, k].tolist()
            df_content[f"predicted_{col_name}"] = _preds_arr[:, k].tolist()

        pd.DataFrame(df_content).to_csv(
            Path(self.output_path, f"{model_name}_fit_data").with_suffix(".csv"),
            index=True,
        )

        return self.model

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict(features)