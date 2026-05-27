from __future__ import annotations

from copy import deepcopy
import logging

import numpy as np
from deap import creator, gp, tools
from scipy.optimize import minimize


logger = logging.getLogger(__name__)

# Penalty used when evaluation fails.  The tuple length is determined
# dynamically by ``make_penalty_fitness``.
_PENALTY_VALUE = 1e18


def make_penalty_fitness(n_targets: int) -> tuple[float, ...]:
    """Return a penalty fitness tuple for *n_targets* targets.

    Shape: ``(mse_t0, mse_t1, ..., mse_t_{n-1}, complexity)``
    """
    return (_PENALTY_VALUE,) * (n_targets + 1)


# ---------------------------------------------------------------------------
# Primitive-set construction helpers
# ---------------------------------------------------------------------------


def build_seed_individuals(
    pset: gp.PrimitiveSet,
    n_features: int,
) -> list[gp.PrimitiveTree]:
    """Create hand-crafted seed individuals that use multiple features.

    These provide the GP with multi-variable starting structures that it can
    refine via crossover, mutation, and constant optimisation.  The seed
    individuals are target-count agnostic — they describe tree *structure*
    only; fitness is assigned by the caller.
    """
    prim: dict[str, gp.Primitive] = {
        p.name: p for prims in pset.primitives.values() for p in prims
    }
    term: dict[str, gp.Terminal] = {
        t.name: t for terms in pset.terminals.values() for t in terms
    }

    add_prim = prim["_add"]
    mul_prim = prim["_mul"]

    def _arg(i: int) -> gp.Terminal:
        return term[f"ARG{i}"]

    def _const(v: float) -> gp.Terminal:
        return make_constant_terminal(v)

    const_zero = _const(0.0)

    seeds: list[list[gp.Primitive | gp.Terminal]] = []

    # 1. Linear: c1*ARGi + c2*ARGj  (all pairs)
    for i in range(n_features):
        for j in range(i + 1, n_features):
            tokens = [
                add_prim,
                mul_prim,
                _const(1.0),
                _arg(i),
                mul_prim,
                _const(1.0),
                _arg(j),
            ]
            seeds.append(tokens)

    # 2. Full linear: c0 + c1*ARG0 + c2*ARG1 + ...
    if n_features >= 2:
        tokens = [mul_prim, _const(1.0), _arg(0)]
        for i in range(1, n_features):
            tokens = [add_prim] + tokens + [mul_prim, _const(1.0), _arg(i)]
        tokens = [add_prim, const_zero] + tokens
        seeds.append(tokens)

    # 3. Quadratic in main feature + linear in other
    if "_square" in prim:
        square_prim = prim["_square"]
        for main in range(min(n_features, 2)):
            other = 1 - main
            tokens = [
                add_prim,
                add_prim,
                mul_prim,
                _const(1.0),
                _arg(main),
                mul_prim,
                _const(1.0),
                square_prim,
                _arg(main),
                mul_prim,
                _const(1.0),
                _arg(other),
            ]
            seeds.append(tokens)

    # 4. Product interaction: c * ARG0 * ARG1
    if n_features >= 2:
        tokens = [mul_prim, _const(1.0), mul_prim, _arg(0), _arg(1)]
        seeds.append(tokens)

    # 5. Ratio: ARG0 / ARG1
    if n_features >= 2:
        tokens = [prim["_protected_div"], _arg(0), _arg(1)]
        seeds.append(tokens)

    individuals: list[gp.PrimitiveTree] = []
    for token_list in seeds:
        try:
            ind = creator.SymbolicIndividual(token_list)  # type: ignore[attr-defined]
            individuals.append(ind)
        except Exception:
            continue

    logger.debug(
        "Seed individuals created",
        extra={"count": len(individuals), "n_features": n_features},
    )
    return individuals


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def vectorised_evaluate(func: object, features: np.ndarray) -> np.ndarray:
    try:
        n_cols = features.shape[1]
        columns = [features[:, i] for i in range(n_cols)]
        result = func(*columns)
        if result is None:
            return _safe_evaluate_rows(func, features)
        arr = np.asarray(result, dtype=np.float64)
        if arr.shape != (features.shape[0],):
            arr = np.full(features.shape[0], float(arr), dtype=np.float64)
        if np.all(np.isfinite(arr)):
            return arr
        return _safe_evaluate_rows(func, features)
    except Exception:
        return _safe_evaluate_rows(func, features)


def _safe_evaluate_rows(func: object, features: np.ndarray) -> np.ndarray:
    """Row-by-row fallback returning NaN on failures."""
    results = np.empty(features.shape[0], dtype=float)
    for i, row in enumerate(features):
        try:
            value = func(*row)  # type: ignore[operator]
            results[i] = np.nan if value is None else float(value)
        except TypeError, ValueError, ZeroDivisionError, OverflowError:
            results[i] = np.nan
    return results


def make_constant_terminal(value: float) -> gp.Terminal:
    """Create a constant terminal with ``object`` return type."""
    return gp.Terminal(float(value), False, object)


_COMPILE_CACHE: dict[str, object] = {}
_COMPILE_CACHE_MAX = 20_000


def _compile_cached(individual: gp.PrimitiveTree, pset: gp.PrimitiveSet) -> object:
    key = str(individual)
    func = _COMPILE_CACHE.get(key)
    if func is None:
        func = gp.compile(individual, pset)
        if len(_COMPILE_CACHE) < _COMPILE_CACHE_MAX:
            _COMPILE_CACHE[key] = func
    return func


def _mse_per_target(
    preds: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """Compute per-column MSE.

    Parameters
    ----------
    preds:
        Shape ``(n_samples,)`` — the GP tree produces a single output.
    targets:
        Shape ``(n_samples,)`` for single-target or
        ``(n_samples, n_targets)`` for multi-target.

    Returns
    -------
    np.ndarray
        Shape ``(n_targets,)`` with one MSE value per target column.
    """
    if targets.ndim == 1:
        diff = preds - targets
        return np.array([float(np.dot(diff, diff) / len(diff))])

    # Multi-target: each target column gets its own MSE against the same
    # single GP prediction.  This is the "shared expression" formulation —
    # the regressor finds one symbolic formula that fits all targets
    # simultaneously and NSGA-II balances the per-target trade-offs.
    n_samples, n_targets = targets.shape
    mses = np.empty(n_targets, dtype=np.float64)
    for k in range(n_targets):
        diff = preds - targets[:, k]
        mses[k] = float(np.dot(diff, diff) / n_samples)
    return mses


def evaluate_individual(
    individual: gp.PrimitiveTree,
    pset: gp.PrimitiveSet,
    features: np.ndarray,
    targets: np.ndarray,
    parsimony_coefficient: float = 0.0,
) -> tuple[float, ...]:
    """Evaluate fitness as ``(mse_t0, ..., mse_t_{n-1}, complexity)``.

    For single-target problems the tuple is ``(penalised_mse, complexity)``,
    which is identical to the previous two-objective interface.

    For multi-target problems each target gets its own MSE objective, and
    complexity is appended as the final objective.  NSGA-II then handles the
    full Pareto trade-off across all objectives.

    Parameters
    ----------
    individual:
        The GP tree to evaluate.
    pset:
        DEAP primitive set.
    features:
        Shape ``(n_samples, n_features)``.
    targets:
        Shape ``(n_samples,)`` or ``(n_samples, n_targets)``.
    parsimony_coefficient:
        Weight of the complexity penalty added to *each* MSE objective.
        For multi-target problems the same coefficient is applied uniformly
        so that the Pareto selection pressure is consistent across targets.
    """
    n_targets = 1 if targets.ndim == 1 else targets.shape[1]
    penalty = make_penalty_fitness(n_targets)

    try:
        func = gp.compile(individual, pset)
        preds = vectorised_evaluate(func, features)
        if not np.all(np.isfinite(preds)):
            return penalty
        mses = _mse_per_target(preds, targets)
    except Exception:
        return penalty

    complexity = float(len(individual))
    penalty_term = parsimony_coefficient * complexity
    # Each MSE objective gets the same parsimony penalty so selection
    # pressure is symmetric across targets.
    penalised = tuple(float(m) + penalty_term for m in mses)
    return penalised + (complexity,)


def optimize_constants(
    individual: gp.PrimitiveTree,
    pset: gp.PrimitiveSet,
    features: np.ndarray,
    targets: np.ndarray,
) -> None:
    """Numerically optimise the ephemeral constants in *individual* in-place.

    Minimises the *mean* MSE across all targets so that the optimiser treats
    every target equally.  This is intentionally a scalar objective (Nelder-Mead
    does not handle vectors) — the multi-objective trade-off is left to NSGA-II.
    """
    indices = [
        idx
        for idx, node in enumerate(individual)
        if isinstance(node, gp.Terminal) and isinstance(node.value, (int, float))
    ]
    if not indices:
        return

    initial = np.array([float(individual[idx].value) for idx in indices], dtype=float)
    n_samples = targets.shape[0]
    n_targets = 1 if targets.ndim == 1 else targets.shape[1]
    inv_n = 1.0 / n_samples

    def objective(constants: np.ndarray) -> float:
        for idx, value in zip(indices, constants, strict=False):
            individual[idx] = make_constant_terminal(float(value))
        try:
            func = _compile_cached(individual, pset)
            preds = vectorised_evaluate(func, features)
        except Exception:
            return 1e18
        if not np.isfinite(preds).all():
            return 1e18
        # Mean MSE across targets
        if n_targets == 1:
            _targets_1d = targets if targets.ndim == 1 else targets[:, 0]
            diff = preds - _targets_1d
            return float(np.dot(diff, diff) * inv_n)
        total = 0.0
        for k in range(n_targets):
            diff = preds - targets[:, k]
            total += float(np.dot(diff, diff) * inv_n)
        return total / n_targets

    initial_mse = objective(initial)
    maxiter = min(80, 15 + 8 * len(indices))

    result = minimize(
        objective,
        initial,
        method="Nelder-Mead",
        options={"maxiter": maxiter, "xatol": 1e-5, "fatol": 1e-7},
    )
    best = result.x if result.fun < initial_mse else initial
    for idx, value in zip(indices, best, strict=False):
        individual[idx] = make_constant_terminal(float(value))


# ---------------------------------------------------------------------------
# Island migration
# ---------------------------------------------------------------------------


def migrate(
    islands: list[list[gp.PrimitiveTree]],
    migration_size: int,
    rng: np.random.Generator,
) -> None:
    """Ring-topology migration: each island sends its best to the next island."""
    n = len(islands)
    if n < 2:
        return

    emigrants: list[list[gp.PrimitiveTree]] = []
    for island in islands:
        best = tools.selBest(island, k=min(migration_size, len(island)))
        emigrants.append([deepcopy(ind) for ind in best])

    for i in range(n):
        dest = (i + 1) % n
        dest_island = islands[dest]
        k = min(migration_size, len(dest_island))
        worst = tools.selWorst(dest_island, k=k)
        worst_set = {id(w) for w in worst}
        islands[dest] = [ind for ind in dest_island if id(ind) not in worst_set]
        islands[dest].extend(emigrants[i])

    logger.debug(
        "Migration complete",
        extra={"islands": n, "migrants_per_island": migration_size},
    )
