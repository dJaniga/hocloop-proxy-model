from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from deap import base, gp, tools
from tqdm import tqdm

from .algebraic_simplification import simplify_island
from .helpers import (
    _PENALTY_VALUE,
    build_seed_individuals,
    evaluate_individual,
    migrate,
    optimize_constants,
    vectorised_evaluate,
)
from .primitives import build_primitive_set
from .toolbox import build_toolbox
from .. import Regressor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _shallow_clone(ind: gp.PrimitiveTree) -> gp.PrimitiveTree:
    """Shallow-clone an individual: the node list is copied, fitness preserved."""
    new = ind.__class__(list(ind))
    if ind.fitness.valid:  # type: ignore[attr-defined]
        new.fitness.values = ind.fitness.values  # type: ignore[attr-defined]
    return new


def _tree_key(ind: gp.PrimitiveTree) -> tuple:
    """Hashable structural fingerprint of an individual."""
    out: list = []
    for node in ind:
        if isinstance(node, gp.Terminal):
            out.append(("T", node.name, node.value))
        else:
            out.append(("P", node.name))
    return tuple(out)


def _mean_mse(fitness_values: tuple[float, ...]) -> float:
    """Mean of all MSE objectives (everything except the last complexity value)."""
    return float(np.mean(fitness_values[:-1]))


def _best_by_mean_mse(
    population: list[gp.PrimitiveTree],
) -> gp.PrimitiveTree | None:
    """Pick the individual with the lowest mean MSE across all targets."""
    if not population:
        return None
    valid = [
        ind
        for ind in population
        if ind.fitness.valid  # type: ignore[attr-defined]
        and all(v < _PENALTY_VALUE for v in ind.fitness.values[:-1])  # type: ignore[attr-defined]
    ]
    candidates = valid if valid else population
    return min(candidates, key=lambda ind: _mean_mse(ind.fitness.values))  # type: ignore[attr-defined]


def _best_by_mean_mse_across_islands(
    islands: list[list[gp.PrimitiveTree]],
) -> gp.PrimitiveTree | None:
    best: gp.PrimitiveTree | None = None
    best_mean = float("inf")
    for island in islands:
        for ind in island:
            if not ind.fitness.valid:  # type: ignore[attr-defined]
                continue
            mmse = _mean_mse(ind.fitness.values)  # type: ignore[attr-defined]
            if mmse < best_mean and all(
                v < _PENALTY_VALUE for v in ind.fitness.values[:-1]
            ):  # type: ignore[attr-defined]
                best_mean = mmse
                best = ind
    return best


# ---------------------------------------------------------------------------
# Process-pool worker (module-level for picklability)
# ---------------------------------------------------------------------------


def _evolve_island_worker(args: tuple) -> list[gp.PrimitiveTree]:
    """Worker for ProcessPoolExecutor — must be defined at module level."""
    (
        island,
        island_size,
        features_std,
        targets_std,
        pset,
        max_tree_height,
        tournament_size,
        crossover_rate,
        mutation_rate,
        const_opt_top_k_ratio,
        parsimony_coefficient,
        fitness_cache,
        rng_seed,
        n_targets,
    ) = args

    toolbox = build_toolbox(
        pset,
        max_tree_height=max_tree_height,
        tournament_size=tournament_size,
        n_targets=n_targets,
    )
    rng = np.random.default_rng(rng_seed)

    def _eval_cached(ind: gp.PrimitiveTree) -> tuple[float, ...]:
        key = _tree_key(ind)
        if key in fitness_cache:
            return fitness_cache[key]
        fit = evaluate_individual(
            ind, pset, features_std, targets_std, parsimony_coefficient
        )
        fitness_cache[key] = fit
        return fit

    offspring = toolbox.select(island, len(island))
    offspring = [_shallow_clone(ind) for ind in offspring]

    for c1, c2 in zip(offspring[::2], offspring[1::2]):
        if rng.random() < crossover_rate:
            toolbox.mate(c1, c2)
            if c1.fitness.valid:
                del c1.fitness.values
            if c2.fitness.valid:
                del c2.fitness.values

    for mutant in offspring:
        if rng.random() < mutation_rate:
            toolbox.mutate(mutant)
            if mutant.fitness.valid:
                del mutant.fitness.values

    for child in offspring:
        if not child.fitness.valid:
            child.fitness.values = _eval_cached(child)

    survivors = tools.selNSGA2(island + offspring, island_size)

    top_k = max(1, int(island_size * const_opt_top_k_ratio))
    elite = sorted(survivors, key=lambda i: _mean_mse(i.fitness.values))[:top_k]
    for ind in elite:
        optimize_constants(ind, pset, features_std, targets_std)
        ind.fitness.values = _eval_cached(ind)

    return survivors


# ---------------------------------------------------------------------------
# Main regressor
# ---------------------------------------------------------------------------


@dataclass
class SymbolicRegressor(Regressor):
    """Hybrid symbolic regressor: GP + NSGA-II + island migration + SymPy simplification.

    Supports **single-target** and **multi-target** regression with the same
    interface.  When ``targets`` passed to :meth:`fit` has shape
    ``(n_samples, n_targets)``, the fitness vector becomes
    ``(mse_t0, mse_t1, ..., mse_t_{n-1}, complexity)`` and NSGA-II optimises
    all objectives simultaneously.  ``predict`` then returns an array of shape
    ``(n_samples, n_targets)`` (or ``(n_samples,)`` for single-target).

    The single symbolic tree that is learnt represents a *shared expression*:
    one formula whose output is compared against every target.  This is
    appropriate when the targets are related and a common functional form is
    expected (e.g. different sensor channels measuring the same phenomenon).
    """

    population_size: int = 200
    generations: int = 100
    mutation_rate: float = 0.2
    crossover_rate: float = 0.7
    tournament_size: int = 3
    max_tree_height: int = 6
    tolerance: float = 1e-4
    seed: int | None = None
    n_islands: int = 4
    migration_interval: int = 5
    migration_size: int = 3
    simplify_interval: int = 10
    parsimony_coefficient: float = 0.001
    basic_arithmetic_only: bool = False
    features_name: tuple[str, ...] | None = None
    targets_name: tuple[str, ...] | None = None
    const_opt_top_k_ratio: float = 0.25
    parallel_islands: bool = True

    # --- post-fit state (not constructor args) ---
    pareto_front_: list[gp.PrimitiveTree] = field(default_factory=list)
    best_individual_: gp.PrimitiveTree | None = None
    _toolbox: base.Toolbox | None = None
    _pset: gp.PrimitiveSet | None = None
    _feature_mean: np.ndarray | None = None
    _feature_std: np.ndarray | None = None
    # Scalars for single-target; arrays of shape (n_targets,) for multi-target
    _target_mean: np.ndarray | float = 0.0
    _target_std: np.ndarray | float = 1.0
    _n_targets: int = 1
    _fitness_cache: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return "symbolic_regressor"

    # ---- standardization ---------------------------------------------------

    def _standardize_features(
        self, features: np.ndarray, *, fit: bool = False
    ) -> np.ndarray:
        if fit:
            self._feature_mean = features.mean(axis=0)
            _std = features.std(axis=0)
            _std[_std < 1e-12] = 1.0
            self._feature_std = _std
        assert self._feature_mean is not None and self._feature_std is not None
        return (features - self._feature_mean) / self._feature_std

    def _standardize_targets(
        self, targets: np.ndarray, *, fit: bool = False
    ) -> np.ndarray:
        """Standardize targets, handling both 1-D (single) and 2-D (multi) arrays.

        Returns an array of the same shape as *targets* with zero mean and unit
        variance per column.  When ``fit=True`` the mean/std statistics are
        stored for later use by :meth:`predict`.
        """
        targets = np.asarray(targets, dtype=np.float64)
        if targets.ndim == 1:
            targets = targets.reshape(-1, 1)

        if fit:
            self._n_targets = targets.shape[1]
            self._target_mean = targets.mean(axis=0)  # shape (n_targets,)
            _std = targets.std(axis=0)
            _std[_std < 1e-12] = 1.0
            self._target_std = _std  # shape (n_targets,)

        targets_std = (targets - self._target_mean) / self._target_std
        # Squeeze back to 1-D for single-target so downstream code is unchanged
        return targets_std.squeeze(axis=1) if self._n_targets == 1 else targets_std

    def _unstandardize_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """Invert target standardization.

        For single-target returns shape ``(n_samples,)``.
        For multi-target, *predictions* is ``(n_samples,)`` (one GP output) and
        is broadcast against each target's mean/std, returning
        ``(n_samples, n_targets)``.
        """
        if self._n_targets == 1:
            return predictions * self._target_std + self._target_mean  # type: ignore[operator]

        # Broadcast single-column predictions to all target scales
        pred_col = predictions[:, np.newaxis]  # (n_samples, 1)
        return pred_col * self._target_std + self._target_mean  # (n_samples, n_targets)

    # ---- fitness cache -----------------------------------------------------

    def _evaluate_with_cache(
        self,
        ind: gp.PrimitiveTree,
        features: np.ndarray,
        targets: np.ndarray,
    ) -> tuple[float, ...]:
        key = _tree_key(ind)
        cached = self._fitness_cache.get(key)
        if cached is not None:
            return cached
        fit = evaluate_individual(
            ind, self._pset, features, targets, self.parsimony_coefficient
        )
        if len(self._fitness_cache) < 50_000:
            self._fitness_cache[key] = fit
        return fit

    # ---- island evolution --------------------------------------------------

    def _evolve_one_island(
        self,
        island: list[gp.PrimitiveTree],
        island_size: int,
        features_std: np.ndarray,
        targets_std: np.ndarray,
        rng: np.random.Generator,
    ) -> list[gp.PrimitiveTree]:
        toolbox = self._toolbox
        assert toolbox is not None

        offspring = toolbox.select(island, len(island))  # type: ignore[attr-defined]
        offspring = [_shallow_clone(ind) for ind in offspring]

        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if rng.random() < self.crossover_rate:
                toolbox.mate(c1, c2)  # type: ignore[attr-defined]
                if c1.fitness.valid:  # type: ignore[attr-defined]
                    del c1.fitness.values  # type: ignore[attr-defined]
                if c2.fitness.valid:  # type: ignore[attr-defined]
                    del c2.fitness.values  # type: ignore[attr-defined]

        for mutant in offspring:
            if rng.random() < self.mutation_rate:
                toolbox.mutate(mutant)  # type: ignore[attr-defined]
                if mutant.fitness.valid:  # type: ignore[attr-defined]
                    del mutant.fitness.values  # type: ignore[attr-defined]

        for child in offspring:
            if not child.fitness.valid:  # type: ignore[attr-defined]
                child.fitness.values = self._evaluate_with_cache(  # type: ignore[attr-defined]
                    child, features_std, targets_std
                )

        survivors = tools.selNSGA2(island + offspring, island_size)

        top_k = max(1, int(island_size * self.const_opt_top_k_ratio))
        elite = sorted(survivors, key=lambda i: _mean_mse(i.fitness.values))[:top_k]  # type: ignore[attr-defined]
        for ind in elite:
            optimize_constants(ind, self._pset, features_std, targets_std)
            new_fit = self._evaluate_with_cache(ind, features_std, targets_std)
            ind.fitness.values = new_fit  # type: ignore[attr-defined]

        return survivors

    # ---- public API --------------------------------------------------------

    def get_fit_details(self) -> dict[str, Any]:
        if self.best_individual_ is None:
            raise ValueError("Model has not been fit yet.")
        fv = self.best_individual_.fitness.values  # type: ignore[attr-defined]
        details: dict[str, Any] = {
            "pareto_size": len(self.pareto_front_),
            "best_complexity": fv[-1],
            "expression": str(self.best_individual_),
            "n_targets": self._n_targets,
        }
        if self._n_targets == 1:
            details["best_mse"] = fv[0]
        else:
            for k in range(self._n_targets):
                name = self.targets_name[k] if self.targets_name else f"target_{k}"
                details[f"best_mse_{name}"] = fv[k]
            details["best_mean_mse"] = _mean_mse(fv)
        return details

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        features_name: tuple[str, ...] | None = None,
        targets_name: tuple[str, ...] | None = None,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> SymbolicRegressor:
        """Fit the symbolic regressor.

        Parameters
        ----------
        features:
            Training features, shape ``(n_samples, n_features)``.
        targets:
            Training targets.  Shape ``(n_samples,)`` for single-target or
            ``(n_samples, n_targets)`` for multi-target.
        features_name:
            Optional human-readable names for each feature column.
        targets_name:
            Optional human-readable names for each target column.
        eval_set:
            Optional ``(eval_features, eval_targets)`` for validation-based
            early stopping.  Target shape must match *targets*.
        """
        rng = np.random.default_rng(self.seed)
        n_features = features.shape[1]

        features_std = np.ascontiguousarray(
            self._standardize_features(features, fit=True), dtype=np.float64
        )
        targets_std = np.ascontiguousarray(
            self._standardize_targets(targets, fit=True), dtype=np.float64
        )
        n_targets = self._n_targets  # set by _standardize_targets(..., fit=True)

        eval_features_std: np.ndarray | None = None
        eval_targets_std: np.ndarray | None = None
        if eval_set is not None:
            eval_features_std = np.ascontiguousarray(
                self._standardize_features(eval_set[0], fit=False), dtype=np.float64
            )
            eval_t = np.asarray(eval_set[1], dtype=np.float64)
            eval_targets_std = np.ascontiguousarray(
                (eval_t - self._target_mean) / self._target_std, dtype=np.float64
            )

        # ---- build pset / toolbox ------------------------------------------
        self._pset = build_primitive_set(n_features, self.basic_arithmetic_only)
        self.features_name = tuple(
            features_name if features_name else (f"ARG{i}" for i in range(n_features))
        )
        self._pset.renameArguments(
            **{f"ARG{idx}": name for idx, name in enumerate(self.features_name)}
        )
        self.targets_name = (
            tuple(targets_name)
            if targets_name
            else tuple(f"target_{k}" for k in range(n_targets))
        )

        self._toolbox = build_toolbox(
            self._pset,
            max_tree_height=self.max_tree_height,
            tournament_size=self.tournament_size,
            n_targets=n_targets,
        )

        island_size = self.population_size // self.n_islands
        if island_size < 4:
            raise ValueError(
                f"population_size={self.population_size} is too small for "
                f"n_islands={self.n_islands} (need at least 4 per island)."
            )

        # logger.debug(
        #     "Initializing symbolic regression",
        #     extra={
        #         "population": self.population_size,
        #         "generations": self.generations,
        #         "islands": self.n_islands,
        #         "island_size": island_size,
        #         "n_targets": n_targets,
        #         "parsimony_coefficient": self.parsimony_coefficient,
        #     },
        # )

        # ---- initial population --------------------------------------------
        seed_individuals = build_seed_individuals(self._pset, n_features)
        islands: list[list[gp.PrimitiveTree]] = []
        for island_idx in range(self.n_islands):
            island = self._toolbox.population(n=island_size)  # type: ignore[attr-defined]
            if island_idx == 0 and seed_individuals:
                n_inject = min(len(seed_individuals), island_size // 2)
                island[:n_inject] = [
                    _shallow_clone(s) for s in seed_individuals[:n_inject]
                ]

            for ind in island:
                ind.fitness.values = self._evaluate_with_cache(  # type: ignore[attr-defined]
                    ind, features_std, targets_std
                )
            top_k = max(1, int(island_size * self.const_opt_top_k_ratio))
            elite = sorted(island, key=lambda i: _mean_mse(i.fitness.values))[:top_k]  # type: ignore[attr-defined]
            for ind in elite:
                optimize_constants(ind, self._pset, features_std, targets_std)
                ind.fitness.values = self._evaluate_with_cache(  # type: ignore[attr-defined]
                    ind, features_std, targets_std
                )
            island = tools.selNSGA2(island, len(island))
            islands.append(island)

        # ---- main loop -----------------------------------------------------
        best_eval_mean_mse = float("inf")
        patience = max(self.generations // 10, 10)
        patience_counter = 0
        best_islands_snapshot: list[list[gp.PrimitiveTree]] | None = None

        island_rng_seeds = [
            int(rng.integers(0, 2**63 - 1)) for _ in range(self.n_islands)
        ]

        executor: ProcessPoolExecutor | None = None
        if self.parallel_islands and self.n_islands > 1:
            executor = ProcessPoolExecutor(max_workers=self.n_islands)

        try:
            for generation in tqdm(range(1, self.generations + 1)):
                # ---- evolve islands ----------------------------------------
                if executor is not None:
                    worker_args = [
                        (
                            islands[i],
                            island_size,
                            features_std,
                            targets_std,
                            self._pset,
                            self.max_tree_height,
                            self.tournament_size,
                            self.crossover_rate,
                            self.mutation_rate,
                            self.const_opt_top_k_ratio,
                            self.parsimony_coefficient,
                            dict(self._fitness_cache),
                            island_rng_seeds[i],
                            n_targets,
                        )
                        for i in range(self.n_islands)
                    ]
                    islands = list(executor.map(_evolve_island_worker, worker_args))
                    island_rng_seeds = [
                        int(rng.integers(0, 2**63 - 1)) for _ in range(self.n_islands)
                    ]
                else:
                    island_rngs = [np.random.default_rng(s) for s in island_rng_seeds]
                    islands = [
                        self._evolve_one_island(
                            islands[i],
                            island_size,
                            features_std,
                            targets_std,
                            island_rngs[i],
                        )
                        for i in range(self.n_islands)
                    ]
                    island_rng_seeds = [
                        int(rng.integers(0, 2**63 - 1)) for _ in range(self.n_islands)
                    ]

                # ---- periodic simplification (Pareto-front only) -----------
                if (
                    self.simplify_interval > 0
                    and generation % self.simplify_interval == 0
                ):
                    for island in islands:
                        front = tools.sortNondominated(
                            island, len(island), first_front_only=True
                        )[0]
                        simplify_island(
                            front,
                            self._pset,
                            features_std,
                            targets_std,
                            self.parsimony_coefficient,
                            n_features,
                        )

                # ---- periodic migration ------------------------------------
                if (
                    self.n_islands > 1
                    and self.migration_interval > 0
                    and generation % self.migration_interval == 0
                ):
                    migrate(islands, self.migration_size, rng)

                # ---- find current best -------------------------------------
                best = _best_by_mean_mse_across_islands(islands)

                # Train tolerance: check mean MSE across all targets
                if best is not None and _mean_mse(best.fitness.values) < self.tolerance:  # type: ignore[attr-defined]
                    logger.debug(
                        "Early stopping reached (train tolerance)",
                        extra={
                            "generation": generation,
                            "mean_mse": _mean_mse(best.fitness.values),  # type: ignore[attr-defined]
                        },
                    )
                    best_islands_snapshot = islands
                    break

                # Validation early stopping
                if (
                    best is not None
                    and eval_features_std is not None
                    and eval_targets_std is not None
                ):
                    val_fitness = evaluate_individual(
                        best,
                        self._pset,
                        eval_features_std,
                        eval_targets_std,
                        self.parsimony_coefficient,
                    )
                    val_mean_mse = _mean_mse(val_fitness)

                    if val_mean_mse < best_eval_mean_mse:
                        best_eval_mean_mse = val_mean_mse
                        patience_counter = 0
                        best_islands_snapshot = [
                            [deepcopy(ind) for ind in island] for island in islands
                        ]
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            logger.debug(
                                "Early stopping reached (validation patience)",
                                extra={
                                    "generation": generation,
                                    "val_mean_mse": val_mean_mse,
                                    "best_val_mean_mse": best_eval_mean_mse,
                                },
                            )
                            break
                else:
                    best_islands_snapshot = islands

                # logger.debug(
                #     "Generation complete",
                #     extra={
                #         "generation": generation,
                #         "best_mean_mse": _mean_mse(best.fitness.values)
                #         if best
                #         else None,  # type: ignore[attr-defined]
                #         "best_len": len(best) if best else None,
                #         "cache_size": len(self._fitness_cache),
                #     },
                # )
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

        # ---- finalize -------------------------------------------------------
        if best_islands_snapshot is not None and best_islands_snapshot is not islands:
            islands = best_islands_snapshot
        elif best_islands_snapshot is islands:
            islands = [[deepcopy(ind) for ind in isl] for isl in islands]

        all_individuals = [ind for island in islands for ind in island]

        if self.simplify_interval > 0:
            front = tools.sortNondominated(
                all_individuals, len(all_individuals), first_front_only=True
            )[0]
            simplify_island(
                front,
                self._pset,
                features_std,
                targets_std,
                self.parsimony_coefficient,
                n_features,
            )

        self.pareto_front_ = tools.sortNondominated(
            all_individuals, len(all_individuals), first_front_only=True
        )[0]
        self.best_individual_ = _best_by_mean_mse(self.pareto_front_)
        if self.best_individual_ is None:
            raise RuntimeError("Symbolic regression did not produce a valid model.")

        self._fitness_cache.clear()
        # logger.debug("Symbolic regression complete", extra=self.get_fit_details())
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Generate predictions from the best individual found during fit.

        Returns
        -------
        np.ndarray
            Shape ``(n_samples,)`` for single-target or
            ``(n_samples, n_targets)`` for multi-target.
        """
        if self.best_individual_ is None or self._pset is None:
            raise ValueError("Model has not been fit yet.")
        features_std = self._standardize_features(features)
        func = gp.compile(self.best_individual_, self._pset)
        predictions_std = vectorised_evaluate(func, features_std)
        predictions = self._unstandardize_predictions(predictions_std)
        # logger.debug(
        #     "Symbolic prediction complete",
        #     extra={"samples": int(features.shape[0]), "n_targets": self._n_targets},
        # )
        return predictions

    def predict_target(self, features: np.ndarray, target_idx: int) -> np.ndarray:
        """Convenience method: predict and return a single target column.

        Only meaningful for multi-target problems (``n_targets > 1``).

        Parameters
        ----------
        features:
            Shape ``(n_samples, n_features)``.
        target_idx:
            Index of the target column to return (0-based).
        """
        if self._n_targets == 1:
            if target_idx != 0:
                raise IndexError(
                    f"target_idx={target_idx} out of range for single-target model."
                )
            return self.predict(features)
        preds = self.predict(features)  # (n_samples, n_targets)
        return preds[:, target_idx]
