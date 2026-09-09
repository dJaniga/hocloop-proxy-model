"""Hyperparameter optimisation for the structured proxy model.

Tuning happens **inside** :meth:`TunedProxyModel.fit`, using only the data that
method is given.  That single decision is what keeps the reported numbers
honest: every protocol in :mod:`modeling.validation` already refits the model on
each training split, so a model that tunes itself during ``fit`` produces proper
nested cross-validation with no extra machinery and no possibility of the search
seeing the evaluation fold.  Tuning once on the whole design and then
cross-validating the winner would report the score of a model chosen with
knowledge of the test data.

Two things can be searched:

*Learner hyperparameters* -- the genetic-programming budget and pressure, or the
usual knobs of the black-box baselines.

*Pipeline choices* -- the feature space and whether the power-law refinement is
applied.  These are ordinary hyperparameters too, and treating them as such
avoids fixing by hand a choice the data can settle.

The default objective is ``r2_log_score`` rather than raw R-squared, because a
target spanning four orders of magnitude makes raw R-squared a lottery on a few
extreme points, and tuning against a noisy objective mostly fits the noise.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Sequence

import numpy as np
import optuna
from sklearn.model_selection import KFold

from modeling.base import Regressor
from modeling.fit_metrics import run_regression_metrics_per_target
from modeling.learners import build_learner_factory
from modeling.proxy import ProxyConfig, StructuredProxyModel

logger = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


# --------------------------------------------------------------------------- #
# Search spaces                                                                #
# --------------------------------------------------------------------------- #


def _symbolic_space(trial: optuna.Trial) -> dict[str, Any]:
    """Budget and selection pressure for the genetic-programming learners."""
    return {
        "population_size": trial.suggest_int("population_size", 100, 600, step=50),
        "generations": trial.suggest_int("generations", 40, 200, step=20),
        "max_tree_height": trial.suggest_int("max_tree_height", 3, 8),
        "parsimony_coefficient": trial.suggest_float(
            "parsimony_coefficient", 1e-4, 1e-2, log=True
        ),
        "n_islands": trial.suggest_int("n_islands", 1, 5),
        "basic_arithmetic_only": trial.suggest_categorical(
            "basic_arithmetic_only", [True, False]
        ),
    }


def _symbolic_deap_space(trial: optuna.Trial) -> dict[str, Any]:
    space = _symbolic_space(trial)
    space.update(
        {
            "tournament_size": trial.suggest_int("tournament_size", 2, 7),
            "mutation_rate": trial.suggest_float("mutation_rate", 0.05, 0.5),
            "crossover_rate": trial.suggest_float("crossover_rate", 0.4, 0.95),
        }
    )
    return space


def _pysr_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "niterations": trial.suggest_int("niterations", 20, 120, step=20),
        "maxsize": trial.suggest_int("maxsize", 15, 40, step=5),
        "populations": trial.suggest_int("populations", 8, 30),
        "population_size": trial.suggest_int("population_size", 20, 60, step=10),
    }


def _gradient_boosting_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "max_iter": trial.suggest_int("max_iter", 200, 1200, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 127, log=True),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 50),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-6, 1.0, log=True),
    }


def _random_forest_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
        "max_depth": trial.suggest_int("max_depth", 5, 40),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features": trial.suggest_float("max_features", 0.3, 1.0),
    }


def _neural_network_space(trial: optuna.Trial) -> dict[str, Any]:
    width = trial.suggest_categorical("width", [64, 128, 256])
    depth = trial.suggest_int("depth", 2, 4)
    return {
        "mlpregressor__hidden_layer_sizes": tuple([width] * depth),
        "mlpregressor__alpha": trial.suggest_float("alpha", 1e-7, 1e-2, log=True),
        "mlpregressor__learning_rate_init": trial.suggest_float(
            "learning_rate_init", 1e-4, 1e-2, log=True
        ),
        "mlpregressor__max_iter": trial.suggest_int("max_iter", 400, 1600, step=200),
    }


def _polynomial_ridge_space(trial: optuna.Trial) -> dict[str, Any]:
    return {"polynomialfeatures__degree": trial.suggest_int("degree", 2, 4)}


#: Learner name to the function that samples its hyperparameters.  A learner
#: absent from this mapping has nothing to tune and is fitted as configured.
LEARNER_SPACES: dict[str, Callable[[optuna.Trial], dict[str, Any]]] = {
    "symbolic_deap": _symbolic_deap_space,
    "symbolic_residual": _symbolic_space,
    "symbolic_pysr": _pysr_space,
    "gradient_boosting": _gradient_boosting_space,
    "random_forest": _random_forest_space,
    "neural_network": _neural_network_space,
    "polynomial_ridge": _polynomial_ridge_space,
}


def _pipeline_space(trial: optuna.Trial) -> dict[str, Any]:
    """Pipeline choices that are hyperparameters like any other."""
    return {
        "feature_space": trial.suggest_categorical(
            "feature_space", ["log_pi_plus_raw", "log_pi", "raw"]
        ),
        "refine_power_law": trial.suggest_categorical("refine_power_law", [True, False]),
    }


# --------------------------------------------------------------------------- #
# Scoring                                                                      #
# --------------------------------------------------------------------------- #


def _score(
    actual: np.ndarray,
    predicted: np.ndarray,
    target_names: Sequence[str],
    metric: str,
) -> float:
    """Mean of *metric* across targets, higher being better."""
    per_target = run_regression_metrics_per_target(
        actual, predicted, target_names=tuple(target_names)
    )
    values = [
        per_target[name][metric]
        for name in target_names
        if not np.isnan(per_target[name].get(metric, np.nan))
    ]
    if not values:
        return -np.inf
    mean = float(np.mean(values))
    # Error metrics improve as they fall, so negate them for a maximising study.
    if metric.startswith("r2") or metric.startswith("d2") or "explained" in metric:
        return mean
    return -mean


# --------------------------------------------------------------------------- #
# The tuned model                                                              #
# --------------------------------------------------------------------------- #


@dataclass
class TuningSettings:
    """How hard to search, and over what."""

    n_trials: int = 25
    inner_splits: int = 3
    metric: str = "r2_log_score"
    search_pipeline: bool = False
    timeout: float | None = None
    seed: int | None = 0
    #: Overrides applied to every trial, e.g. a reduced budget during tuning.
    fixed_overrides: dict[str, Any] = field(default_factory=dict)


@dataclass
class TunedProxyModel(Regressor):
    """A :class:`StructuredProxyModel` that tunes itself on its training data.

    Because the search runs inside :meth:`fit`, dropping this into any protocol
    in :mod:`modeling.validation` yields nested cross-validation: the outer split
    supplies the training data, the inner split selects hyperparameters, and the
    evaluation fold is never seen by the search.
    """

    learner_name: str
    config: ProxyConfig = field(default_factory=ProxyConfig)
    settings: TuningSettings = field(default_factory=TuningSettings)
    name: str = "tuned_proxy"

    model_: StructuredProxyModel | None = None
    best_params_: dict[str, Any] = field(default_factory=dict)
    best_pipeline_params_: dict[str, Any] = field(default_factory=dict)
    inner_score_: float = float("nan")
    n_trials_completed_: int = 0

    def __str__(self) -> str:
        return f"{self.name}_{self.learner_name}"

    # -- helpers -------------------------------------------------------- #

    def _build(
        self, learner_params: dict[str, Any], pipeline_params: dict[str, Any]
    ) -> StructuredProxyModel:
        overrides = {**self.settings.fixed_overrides, **learner_params}
        factory = build_learner_factory(self.learner_name, **overrides)
        config = replace(self.config, **pipeline_params) if pipeline_params else self.config
        return StructuredProxyModel(learner_factory=factory, config=config)

    # -- fitting -------------------------------------------------------- #

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        features_name: tuple[str, ...] | None = None,
        targets_name: tuple[str, ...] | None = None,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> TunedProxyModel:
        features = np.asarray(features, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.float64)
        if targets.ndim == 1:
            targets = targets.reshape(-1, 1)
        names = tuple(
            targets_name or tuple(f"target_{k}" for k in range(targets.shape[1]))
        )
        feature_names = tuple(
            features_name or tuple(f"ARG{i}" for i in range(features.shape[1]))
        )

        space = LEARNER_SPACES.get(self.learner_name)
        if space is None and not self.settings.search_pipeline:
            logger.info("Nothing to tune for %r; fitting as configured.", self.learner_name)
            self.model_ = self._build({}, {})
            self.model_.fit(features, targets, feature_names, names)
            return self

        folds = KFold(
            n_splits=self.settings.inner_splits,
            shuffle=True,
            random_state=self.settings.seed,
        )
        splits = list(folds.split(features))

        def objective(trial: optuna.Trial) -> float:
            learner_params = space(trial) if space is not None else {}
            pipeline_params = _pipeline_space(trial) if self.settings.search_pipeline else {}

            fold_scores: list[float] = []
            for index, (train_index, valid_index) in enumerate(splits):
                candidate = self._build(learner_params, pipeline_params)
                try:
                    candidate.fit(
                        features[train_index], targets[train_index], feature_names, names
                    )
                    predicted = np.asarray(
                        candidate.predict(features[valid_index]), dtype=np.float64
                    )
                except Exception as error:
                    # A hyperparameter combination that cannot be fitted is a
                    # failed trial, not a failed run.
                    logger.debug("Trial failed to fit: %s", error)
                    raise optuna.TrialPruned() from error
                if predicted.ndim == 1:
                    predicted = predicted.reshape(-1, 1)
                fold_scores.append(
                    _score(targets[valid_index], predicted, names, self.settings.metric)
                )
                trial.report(float(np.mean(fold_scores)), index)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return float(np.mean(fold_scores))

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.settings.seed),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        )
        logger.info(
            "Tuning %r: %d trials, %d inner folds, objective %s.",
            self.learner_name,
            self.settings.n_trials,
            self.settings.inner_splits,
            self.settings.metric,
        )
        study.optimize(
            objective,
            n_trials=self.settings.n_trials,
            timeout=self.settings.timeout,
            catch=(Exception,),
        )

        completed = [t for t in study.trials if t.value is not None]
        self.n_trials_completed_ = len(completed)
        if not completed:
            logger.warning(
                "Every tuning trial failed for %r; falling back to the configured "
                "hyperparameters.",
                self.learner_name,
            )
            self.model_ = self._build({}, {})
            self.model_.fit(features, targets, feature_names, names)
            return self

        raw = dict(study.best_params)
        pipeline_keys = {"feature_space", "refine_power_law"}
        self.best_pipeline_params_ = {k: v for k, v in raw.items() if k in pipeline_keys}
        self.inner_score_ = float(study.best_value)

        # Re-sample the winning trial through the space functions so derived
        # parameters (such as the hidden-layer tuple built from width and depth)
        # are reconstructed exactly as they were during the trial.
        best_trial = study.best_trial
        fixed = optuna.trial.FixedTrial(best_trial.params)
        learner_params = space(fixed) if space is not None else {}
        pipeline_params = _pipeline_space(fixed) if self.settings.search_pipeline else {}
        self.best_params_ = {"learner": learner_params, "pipeline": pipeline_params}

        logger.info(
            "Best inner %s = %.4f over %d completed trials; params %s",
            self.settings.metric,
            self.inner_score_,
            self.n_trials_completed_,
            best_trial.params,
        )

        self.model_ = self._build(learner_params, pipeline_params)
        self.model_.fit(features, targets, feature_names, names)
        return self

    # -- delegation ----------------------------------------------------- #

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise ValueError("Model has not been fit yet.")
        return self.model_.predict(features)

    def get_fit_details(self) -> dict[str, Any]:
        if self.model_ is None:
            raise ValueError("Model has not been fit yet.")
        details = self.model_.get_fit_details()
        details["tuning"] = {
            "learner": self.learner_name,
            "metric": self.settings.metric,
            "inner_score": self.inner_score_,
            "n_trials_completed": self.n_trials_completed_,
            "best_params": self.best_params_,
        }
        return details

    def closed_form(self) -> dict[str, str]:
        if self.model_ is None:
            raise ValueError("Model has not been fit yet.")
        return self.model_.closed_form()

    @property
    def structure_(self):  # noqa: D401 - passthrough for downstream consumers
        """The fitted target structure, for callers that reach through."""
        return None if self.model_ is None else self.model_.structure_

    @property
    def target_models_(self):
        return {} if self.model_ is None else self.model_.target_models_

    def prepare_columns(self, features: np.ndarray) -> dict[str, np.ndarray]:
        if self.model_ is None:
            raise ValueError("Model has not been fit yet.")
        return self.model_.prepare_columns(features)


def tuned_model_factory(
    learner_name: str,
    config: ProxyConfig,
    settings: TuningSettings,
) -> Callable[[], TunedProxyModel]:
    """A zero-argument factory, as the validation protocols expect."""

    def factory() -> TunedProxyModel:
        return TunedProxyModel(
            learner_name=learner_name, config=config, settings=settings
        )

    return factory
