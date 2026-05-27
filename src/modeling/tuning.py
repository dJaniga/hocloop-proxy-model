from __future__ import annotations

import copy
import logging
from typing import Sequence

import numpy as np
import optuna
from sklearn.model_selection import KFold

from modeling import Regressor
from modeling.symbolic import SymbolicRegressor
from modeling.tuning_metrics import evaluate_metric, get_metric_direction, MAXIMIZE_METRICS

logger = logging.getLogger(__name__)


def _n_targets_from(targets: np.ndarray) -> int:
    return targets.shape[1] if targets.ndim == 2 and targets.shape[1] > 1 else 1


def _per_target_mse(y_val: np.ndarray, preds: np.ndarray, n_targets: int) -> list[float]:
    """Return a list of per-target MSE values, one per target column."""
    mses = []
    for k in range(n_targets):
        y_k = y_val[:, k] if n_targets > 1 else (y_val[:, 0] if y_val.ndim == 2 else y_val)
        p_k = preds[:, k] if n_targets > 1 else (preds[:, 0] if preds.ndim == 2 else preds)
        diff = y_k - p_k
        mses.append(float(np.dot(diff, diff) / len(diff)))
    return mses


def _select_best_trial_multiobjective(
    trials: Sequence[optuna.trial.FrozenTrial],
    n_targets: int,
) -> optuna.trial.FrozenTrial:
    """Pick the best trial from a Pareto front using mean MSE across target objectives.

    The objective vector is ``(mse_t0, ..., mse_t_{n-1}, complexity)`` — all
    minimised.  Target objectives are indices ``0 .. n_targets-1``; complexity
    is the last index.  We select the trial with the lowest mean across the
    target objectives, which mirrors ``_best_by_mean_mse`` in the regressor.
    """
    return min(
        trials,
        key=lambda t: float(np.mean(t.values[:n_targets])),  # type: ignore[index]
    )


def tune_hyperparameters(
        model: Regressor,
        features: np.ndarray,
        targets: np.ndarray,
        features_name: tuple[str, ...],
        targets_name: tuple[str, ...] | None,
        n_trials: int = 50,
        n_splits: int = 3,
        tuning_metric: str = "mean_squared_error",
        seed: int | None = None,
        n_jobs: int | None = None,
) -> Regressor:
    if not isinstance(model, SymbolicRegressor):
        logger.warning(
            f"Hyperparameter tuning not implemented for {type(model).__name__}. "
            "Returning original model."
        )
        return model

    n_targets = _n_targets_from(targets)
    is_multiobjective = n_targets > 1

    # ------------------------------------------------------------------
    # Objective function
    # ------------------------------------------------------------------
    # Multi-target: returns a tuple (mse_t0_cv, ..., mse_t_{n-1}_cv, complexity_cv)
    #   mirroring the regressor's NSGA-II fitness vector so Optuna's Pareto
    #   front is comparable to the GP Pareto front.
    # Single-target: returns a single float (the CV mean of tuning_metric),
    #   unchanged from the previous behaviour.
    # ------------------------------------------------------------------

    def objective(trial: optuna.Trial) -> tuple[float, ...] | float:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

        # Accumulators: one list per target MSE + one for complexity
        cv_mses: list[list[float]] = [[] for _ in range(n_targets)]
        cv_complexities: list[float] = []
        # Single-target only: arbitrary tuning_metric accumulator
        cv_scalar: list[float] = []

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(features)):
            logger.debug(
                f"Inner CV fold {fold_idx + 1}/{n_splits} — "
                f"{type(model).__name__} hyperparameter tuning"
            )
            X_train, X_val = features[train_idx], features[val_idx]
            y_train, y_val = targets[train_idx], targets[val_idx]

            trial_model = copy.deepcopy(model)

            if isinstance(trial_model, SymbolicRegressor):
                if isinstance(trial_model, SymbolicRegressor):
                    trial_model.population_size = trial.suggest_int(
                        "population_size", 50, 500, step=50
                    )
                    trial_model.generations = trial.suggest_int(
                        "generations", 20, 200, step=10
                    )
                    trial_model.mutation_rate = trial.suggest_float(
                        "mutation_rate", 0.05, 0.5
                    )
                    trial_model.crossover_rate = trial.suggest_float(
                        "crossover_rate", 0.4, 0.95
                    )
                    trial_model.tournament_size = trial.suggest_int(
                        "tournament_size", 2, 10
                    )
                    trial_model.max_tree_height = trial.suggest_int(
                        "max_tree_height", 2, 12
                    )
                    trial_model.n_islands = trial.suggest_int(
                        "n_islands", 1, 8
                    )
                    trial_model.migration_interval = trial.suggest_int(
                        "migration_interval", 2, 20
                    )
                    trial_model.migration_size = trial.suggest_int(
                        "migration_size", 1, 10
                    )
                    trial_model.simplify_interval = trial.suggest_int(
                        "simplify_interval", 5, 30, step=5
                    )
                    trial_model.parsimony_coefficient = trial.suggest_float(
                        "parsimony_coefficient", 0.0001, 0.01, log=True
                    )
                    trial_model.basic_arithmetic_only = trial.suggest_categorical(
                        "basic_arithmetic_only", [True, False]
                    )
                    trial_model.const_opt_top_k_ratio = trial.suggest_float(
                        "const_opt_top_k_ratio", 0.1, 0.5
                    )

            trial_model.fit(
                X_train, y_train,
                features_name=features_name,
                targets_name=targets_name,
                eval_set=(X_val, y_val),
            )
            preds = trial_model.predict(X_val)

            # Per-target MSE — mirrors the regressor's evaluate_individual output
            fold_mses = _per_target_mse(y_val, preds, n_targets)
            for k, mse in enumerate(fold_mses):
                cv_mses[k].append(mse)

            # Complexity: tree node count from the best individual found, exactly
            # as the regressor stores it in fitness.values[-1].
            if (
                isinstance(trial_model, SymbolicRegressor)
                and trial_model.best_individual_ is not None
            ):
                cv_complexities.append(
                    float(trial_model.best_individual_.fitness.values[-1])  # type: ignore[attr-defined]
                )

            # Single-target scalar metric (used only when n_targets == 1)
            if not is_multiobjective:
                y_1d = y_val[:, 0] if y_val.ndim == 2 else y_val
                p_1d = preds[:, 0] if preds.ndim == 2 else preds
                cv_scalar.append(evaluate_metric(tuning_metric, y_1d, p_1d))

        if is_multiobjective:
            # Objective vector: CV-mean MSE per target + CV-mean complexity
            mean_mses = tuple(float(np.mean(cv_mses[k])) for k in range(n_targets))
            mean_complexity = float(np.mean(cv_complexities)) if cv_complexities else 0.0
            result = mean_mses + (mean_complexity,)
            logger.debug(
                "Multi-objective CV result",
                extra={"mean_mses": mean_mses, "mean_complexity": mean_complexity},
            )
            return result
        else:
            mean_score = float(np.mean(cv_scalar))
            logger.debug(f"CV score: {mean_score}")
            return mean_score

    # ------------------------------------------------------------------
    # Study construction
    # ------------------------------------------------------------------
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if is_multiobjective:
        # n_targets MSE objectives + 1 complexity objective, all minimised.
        directions = ["minimize"] * (n_targets + 1)
        sampler = optuna.samplers.NSGAIISampler(seed=seed)
        study = optuna.create_study(directions=directions, sampler=sampler)
        logger.debug(
            f"Starting multi-objective hyperparameter tuning for "
            f"{type(model).__name__} — {n_targets} MSE objectives + complexity, "
            f"{n_trials} trials."
        )
    else:
        direction = get_metric_direction(tuning_metric)
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction=direction, sampler=sampler)
        logger.debug(
            f"Starting hyperparameter tuning for {type(model).__name__} with "
            f"{n_trials} trials, optimizing {tuning_metric} ({direction})."
        )

    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

    # ------------------------------------------------------------------
    # Extract best hyperparameters
    # ------------------------------------------------------------------
    if is_multiobjective:
        # study.best_trials is the full Pareto front — pick using mean MSE
        # across target objectives, mirroring _best_by_mean_mse in the regressor.
        pareto_trials = study.best_trials
        if not pareto_trials:
            logger.warning("No Pareto-optimal trials found; returning original model.")
            return model
        best_trial = _select_best_trial_multiobjective(pareto_trials, n_targets)
        best_mean_mse = float(np.mean(best_trial.values[:n_targets]))  # type: ignore[index]
        logger.debug(
            f"Best Pareto trial — mean MSE across targets: {best_mean_mse:.6f}, "
            f"complexity: {best_trial.values[n_targets]:.1f}, "  # type: ignore[index]
            f"params: {best_trial.params}"
        )
    else:
        best_trial = study.best_trial
        logger.debug(
            f"Best hyperparameters found: {best_trial.params} "
            f"({tuning_metric}={best_trial.value:.6f})"
        )

    best_params = best_trial.params

    best_model = copy.deepcopy(model)
    if isinstance(best_model, SymbolicRegressor):
        best_model.population_size = best_params["population_size"]
        best_model.generations = best_params["generations"]
        best_model.mutation_rate = best_params["mutation_rate"]
        best_model.crossover_rate = best_params["crossover_rate"]
        best_model.tournament_size = best_params["tournament_size"]
        best_model.max_tree_height = best_params["max_tree_height"]
        best_model.n_islands = best_params["n_islands"]
        best_model.migration_interval = best_params["migration_interval"]
        best_model.migration_size = best_params["migration_size"]
        best_model.simplify_interval = best_params["simplify_interval"]
        best_model.parsimony_coefficient = best_params["parsimony_coefficient"]
        best_model.basic_arithmetic_only = best_params["basic_arithmetic_only"]
        best_model.const_opt_top_k_ratio = best_params["const_opt_top_k_ratio"]

    return best_model