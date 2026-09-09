"""Stage-3 learners behind a single scikit-learn-style interface.

Every learner takes a design matrix and a 1-D response and exposes ``fit`` and
``predict``.  Symbolic learners additionally expose ``expression_``, the closed
form they found, and ``pareto_`` where an accuracy versus complexity front is
available.

Holding stages 1, 2 and 4 of :mod:`modeling.proxy` fixed and swapping only the
learner turns the benchmark table into a controlled comparison: differences come
from the regressor, not from a different pipeline around it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Symbolic learners                                                           #
# --------------------------------------------------------------------------- #


@dataclass
class SymbolicLearner:
    """Adapter around the in-house DEAP symbolic regressor.

    Exposes the Pareto front so a downstream selector can trade accuracy against
    expression size instead of always taking the most accurate individual.
    """

    population_size: int = 400
    generations: int = 120
    max_tree_height: int = 6
    parsimony_coefficient: float = 1e-3
    n_islands: int = 4
    tournament_size: int = 3
    mutation_rate: float = 0.25
    crossover_rate: float = 0.7
    basic_arithmetic_only: bool = False
    parallel_islands: bool = False
    seed: int | None = 0
    feature_names: tuple[str, ...] | None = None

    model_: Any = None
    expression_: str | None = None
    pareto_: list[dict[str, Any]] = field(default_factory=list)
    fit_details_: dict[str, Any] = field(default_factory=dict)

    def fit(self, X: np.ndarray, y: np.ndarray) -> SymbolicLearner:
        from modeling.symbolic import SymbolicRegressor

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        names = self.feature_names or tuple(f"x{i}" for i in range(X.shape[1]))

        self.model_ = SymbolicRegressor(
            population_size=self.population_size,
            generations=self.generations,
            max_tree_height=self.max_tree_height,
            parsimony_coefficient=self.parsimony_coefficient,
            n_islands=self.n_islands,
            tournament_size=self.tournament_size,
            mutation_rate=self.mutation_rate,
            crossover_rate=self.crossover_rate,
            basic_arithmetic_only=self.basic_arithmetic_only,
            parallel_islands=self.parallel_islands,
            seed=self.seed,
        )
        self.model_.fit(X, y, features_name=names, targets_name=("response",))
        self.expression_ = str(self.model_.best_individual_)
        self.fit_details_ = self.model_.get_fit_details()
        self.pareto_ = [
            {
                "expression": str(individual),
                "complexity": float(individual.fitness.values[-1]),
                "mse": float(individual.fitness.values[0]),
            }
            for individual in self.model_.pareto_front_
        ]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise ValueError("SymbolicLearner has not been fit yet.")
        return np.asarray(self.model_.predict(np.asarray(X, dtype=np.float64))).ravel()


@dataclass
class PySRLearner:
    """Adapter around PySR, the reference symbolic-regression implementation.

    PySR runs a Julia backend that is installed on first use.  If the backend is
    unavailable the failure is raised at ``fit`` time with the original error, so
    a benchmark run reports a missing dependency rather than silently comparing
    against nothing.
    """

    niterations: int = 60
    population_size: int = 40
    populations: int = 15
    maxsize: int = 30
    binary_operators: tuple[str, ...] = ("+", "-", "*", "/")
    unary_operators: tuple[str, ...] = ("square", "sqrt", "exp", "log")
    seed: int = 0
    procs: int = 0
    feature_names: tuple[str, ...] | None = None
    extra_kwargs: dict[str, Any] = field(default_factory=dict)

    model_: Any = None
    expression_: str | None = None
    pareto_: list[dict[str, Any]] = field(default_factory=list)
    fit_details_: dict[str, Any] = field(default_factory=dict)

    def fit(self, X: np.ndarray, y: np.ndarray) -> PySRLearner:
        from pysr import PySRRegressor

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        names = list(self.feature_names or [f"x{i}" for i in range(X.shape[1])])

        self.model_ = PySRRegressor(
            niterations=self.niterations,
            population_size=self.population_size,
            populations=self.populations,
            maxsize=self.maxsize,
            binary_operators=list(self.binary_operators),
            unary_operators=list(self.unary_operators),
            random_state=self.seed,
            deterministic=self.procs == 0,
            parallelism="serial" if self.procs == 0 else "multiprocessing",
            progress=False,
            verbosity=0,
            **self.extra_kwargs,
        )
        self.model_.fit(X, y, variable_names=names)

        equations = self.model_.equations_
        self.pareto_ = [
            {
                "expression": str(row["equation"]),
                "complexity": float(row["complexity"]),
                "mse": float(row["loss"]),
            }
            for _, row in equations.iterrows()
        ]
        self.expression_ = str(self.model_.sympy())
        self.fit_details_ = {"n_equations": len(self.pareto_)}
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise ValueError("PySRLearner has not been fit yet.")
        return np.asarray(self.model_.predict(np.asarray(X, dtype=np.float64))).ravel()


@dataclass
class PowerLawLearner:
    """Ordinary least squares on the design matrix.

    In the ``log_pi`` feature space this is a pure power law, and it is the
    natural floor for the benchmark: any symbolic model that fails to beat it
    has found nothing that dimensional analysis did not already supply.
    """

    model_: Any = None
    expression_: str | None = None
    feature_names: tuple[str, ...] | None = None
    fit_details_: dict[str, Any] = field(default_factory=dict)

    def fit(self, X: np.ndarray, y: np.ndarray) -> PowerLawLearner:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        self.model_ = LinearRegression().fit(X, y)
        names = self.feature_names or tuple(f"x{i}" for i in range(X.shape[1]))
        terms = [f"{coefficient:+.6g}*{name}" for coefficient, name in zip(self.model_.coef_, names)]
        self.expression_ = f"{self.model_.intercept_:.6g} " + " ".join(terms)
        self.fit_details_ = {
            "coefficients": self.model_.coef_.tolist(),
            "intercept": float(self.model_.intercept_),
        }
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise ValueError("PowerLawLearner has not been fit yet.")
        return np.asarray(self.model_.predict(np.asarray(X, dtype=np.float64))).ravel()


@dataclass
class ResidualSymbolicLearner:
    """Least-squares power law plus a symbolic correction to its residual.

    A genetic search started from random expressions has to rediscover the
    linear part of the relation before it can improve on it, and with ten
    candidate inputs it usually runs out of budget first -- which is why a plain
    symbolic search can score below the ordinary least-squares fit it should
    dominate.  Fitting the residual removes that handicap: the linear fit is
    solved exactly in closed form and the search spends its whole budget on what
    the linear fit cannot express.

    The result is never worse than the linear model by construction, and it
    stays readable.  In the log-Pi space of this pipeline the composition reads
    as a physically meaningful power law multiplied by ``exp(correction)``, so
    the correction is directly interpretable as a departure from power-law
    scaling.
    """

    population_size: int = 400
    generations: int = 120
    max_tree_height: int = 6
    parsimony_coefficient: float = 1e-3
    n_islands: int = 4
    basic_arithmetic_only: bool = False
    parallel_islands: bool = False
    seed: int | None = 0
    feature_names: tuple[str, ...] | None = None

    base_: PowerLawLearner | None = None
    symbolic_: SymbolicLearner | None = None
    expression_: str | None = None
    pareto_: list[dict[str, Any]] = field(default_factory=list)
    fit_details_: dict[str, Any] = field(default_factory=dict)

    def fit(self, X: np.ndarray, y: np.ndarray) -> ResidualSymbolicLearner:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        self.base_ = PowerLawLearner(feature_names=self.feature_names).fit(X, y)
        residual = y - self.base_.predict(X)

        self.symbolic_ = SymbolicLearner(
            population_size=self.population_size,
            generations=self.generations,
            max_tree_height=self.max_tree_height,
            parsimony_coefficient=self.parsimony_coefficient,
            n_islands=self.n_islands,
            basic_arithmetic_only=self.basic_arithmetic_only,
            parallel_islands=self.parallel_islands,
            seed=self.seed,
            feature_names=self.feature_names,
        ).fit(X, residual)

        self.expression_ = f"({self.base_.expression_}) + ({self.symbolic_.expression_})"
        self.pareto_ = self.symbolic_.pareto_
        residual_variance = float(np.var(residual))
        remaining = float(np.var(residual - self.symbolic_.predict(X)))
        self.fit_details_ = {
            "linear": self.base_.fit_details_,
            "symbolic": self.symbolic_.fit_details_,
            "residual_variance_explained": (
                1.0 - remaining / residual_variance if residual_variance > 0 else 0.0
            ),
        }
        logger.info(
            "Residual symbolic learner explained %.1f%% of the linear model's residual variance.",
            100.0 * self.fit_details_["residual_variance_explained"],
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.base_ is None or self.symbolic_ is None:
            raise ValueError("ResidualSymbolicLearner has not been fit yet.")
        X = np.asarray(X, dtype=np.float64)
        return self.base_.predict(X) + self.symbolic_.predict(X)


# --------------------------------------------------------------------------- #
# Black-box baselines                                                         #
# --------------------------------------------------------------------------- #


def gradient_boosting(seed: int | None = 0) -> Any:
    return HistGradientBoostingRegressor(
        max_iter=600,
        learning_rate=0.06,
        max_leaf_nodes=63,
        l2_regularization=1e-3,
        early_stopping=False,
        random_state=seed,
    )


def random_forest(seed: int | None = 0) -> Any:
    return RandomForestRegressor(n_estimators=400, n_jobs=-1, random_state=seed)


def neural_network(seed: int | None = 0) -> Any:
    return make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(256, 256, 128),
            max_iter=1200,
            learning_rate_init=3e-3,
            random_state=seed,
        ),
    )


def polynomial_ridge(degree: int = 3) -> Any:
    return make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=degree, include_bias=False),
        RidgeCV(alphas=np.logspace(-6, 3, 40)),
    )


#: Factories for every learner the benchmark suite compares, keyed by the label
#: used in reports.
LEARNER_FACTORIES: dict[str, Any] = {
    "power_law_ols": lambda: PowerLawLearner(),
    "symbolic_deap": lambda: SymbolicLearner(),
    "symbolic_residual": lambda: ResidualSymbolicLearner(),
    "symbolic_pysr": lambda: PySRLearner(),
    "polynomial_ridge": lambda: polynomial_ridge(),
    "gradient_boosting": lambda: gradient_boosting(),
    "random_forest": lambda: random_forest(),
    "neural_network": lambda: neural_network(),
}


def _apply_override(learner: Any, name: str, key: str, value: Any) -> None:
    """Set one hyperparameter, whichever kind of object the learner is.

    The dataclass learners defined here take plain attributes, while the
    scikit-learn baselines are estimators and pipelines whose nested parameters
    are reachable only through ``set_params`` with a ``step__param`` path.
    Silently ignoring an unknown key would let a tuning run report scores for
    hyperparameters it never actually applied, so an unusable key raises.
    """
    if hasattr(learner, "set_params"):
        try:
            valid = learner.get_params(deep=True)
        except Exception:
            valid = {}
        if key in valid:
            learner.set_params(**{key: value})
            return
    if hasattr(learner, key):
        setattr(learner, key, value)
        return
    raise AttributeError(
        f"{name!r} learner has no hyperparameter {key!r}. "
        f"For a scikit-learn pipeline use a 'step__param' path."
    )


def build_learner_factory(name: str, **overrides: Any) -> Any:
    """Return a zero-argument factory for the named learner.

    Unknown override keys are dropped with a warning rather than raising, so a
    shared set of overrides (a reduced search budget, say) can be handed to
    every learner without the caller having to know which ones accept it.
    Keys the learner does recognise are always applied.
    """
    if name not in LEARNER_FACTORIES:
        raise KeyError(f"Unknown learner {name!r}. Known: {sorted(LEARNER_FACTORIES)}")
    if not overrides:
        return LEARNER_FACTORIES[name]

    base = LEARNER_FACTORIES[name]

    def factory() -> Any:
        learner = base()
        for key, value in overrides.items():
            try:
                _apply_override(learner, name, key, value)
            except AttributeError as error:
                logger.debug("Ignoring override for %r: %s", name, error)
        return learner

    return factory
