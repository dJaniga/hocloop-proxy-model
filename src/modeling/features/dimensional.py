"""Buckingham-Pi reduction of a simulator parameter study.

Given the physical dimension of every feature and of the response, a set of
repeating variables induces a complete basis of dimensionless groups.  Learning
the response in that space has two effects that matter for a proxy model:

* the number of inputs drops by the rank of the dimension matrix, and
* the learnt relation is invariant to the unit system, so it transfers to any
  consistent rescaling of the problem.

Classical dimensional analysis leaves the choice of repeating variables to the
analyst, and different choices give bases of very different practical quality:
the response group of a good basis already absorbs most of the variation of the
response, leaving an easier residual relation to learn.  :func:`select_pi_basis`
therefore enumerates the admissible repeating sets and scores each one on the
data, which turns a modelling convention into a measurable decision.

The basis is arranged so that exactly one group -- the *response group* --
carries the response variable, with unit exponent.  The remaining groups are
pure predictors.  A model learns ``Pi_0 = f(Pi_1, ..., Pi_k)`` and the response
is recovered by multiplying ``Pi_0`` by the monomial that made it
dimensionless.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from fractions import Fraction
from typing import Mapping, Sequence

import numpy as np
import sympy as sp

from modeling.features.units import Dimension, parse_unit

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PiGroup:
    """A dimensionless monomial ``prod(variable ** exponent)``."""

    name: str
    exponents: dict[str, Fraction]

    def formula(self) -> str:
        numerator: list[str] = []
        denominator: list[str] = []
        for variable, exponent in self.exponents.items():
            if exponent == 0:
                continue
            magnitude = abs(exponent)
            rendered = variable if magnitude == 1 else f"{variable}**{magnitude}"
            (numerator if exponent > 0 else denominator).append(rendered)
        top = " * ".join(numerator) if numerator else "1"
        if not denominator:
            return top
        return f"{top} / ({' * '.join(denominator)})"

    def evaluate(self, columns: Mapping[str, np.ndarray]) -> np.ndarray:
        """Evaluate the monomial. Participating columns must be strictly positive."""
        result: np.ndarray | None = None
        for variable, exponent in self.exponents.items():
            if exponent == 0:
                continue
            values = np.asarray(columns[variable], dtype=np.float64)
            term = np.power(values, float(exponent))
            result = term if result is None else result * term
        if result is None:
            return np.ones(len(next(iter(columns.values()))), dtype=np.float64)
        return result

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "formula": self.formula(),
            "exponents": {k: str(v) for k, v in self.exponents.items() if v != 0},
        }


@dataclass
class PiBasis:
    """A complete set of Pi groups for one response variable."""

    response: str
    repeating: tuple[str, ...]
    response_group: PiGroup
    predictor_groups: tuple[PiGroup, ...]

    @property
    def scale_exponents(self) -> dict[str, Fraction]:
        """Exponents of the monomial that non-dimensionalises the response.

        ``response = Pi_0 * prod(variable ** scale_exponent)``, so these are the
        response group's exponents negated, with the response itself removed.
        """
        return {
            variable: -exponent
            for variable, exponent in self.response_group.exponents.items()
            if variable != self.response and exponent != 0
        }

    def scale_formula(self) -> str:
        parts = []
        for variable, exponent in self.scale_exponents.items():
            parts.append(variable if exponent == 1 else f"{variable}**{exponent}")
        return " * ".join(parts) if parts else "1"

    def response_scale(self, columns: Mapping[str, np.ndarray]) -> np.ndarray:
        """The monomial by which ``Pi_0`` must be multiplied to recover units."""
        result: np.ndarray | None = None
        for variable, exponent in self.scale_exponents.items():
            values = np.asarray(columns[variable], dtype=np.float64)
            term = np.power(values, float(exponent))
            result = term if result is None else result * term
        if result is None:
            return np.ones(len(next(iter(columns.values()))), dtype=np.float64)
        return result

    def to_response_group(
        self, columns: Mapping[str, np.ndarray], response_values: np.ndarray
    ) -> np.ndarray:
        """Map response values into the dimensionless response group."""
        return np.asarray(response_values, dtype=np.float64) / self.response_scale(columns)

    def from_response_group(
        self, columns: Mapping[str, np.ndarray], group_values: np.ndarray
    ) -> np.ndarray:
        """Map dimensionless response-group values back to physical units."""
        return np.asarray(group_values, dtype=np.float64) * self.response_scale(columns)

    def transform(self, columns: Mapping[str, np.ndarray]) -> np.ndarray:
        """Predictor groups evaluated as a design matrix."""
        if not self.predictor_groups:
            return np.empty((len(next(iter(columns.values()))), 0), dtype=np.float64)
        return np.column_stack([group.evaluate(columns) for group in self.predictor_groups])

    def predictor_names(self) -> tuple[str, ...]:
        return tuple(group.name for group in self.predictor_groups)

    def to_dict(self) -> dict[str, object]:
        return {
            "response": self.response,
            "repeating_variables": list(self.repeating),
            "response_group": self.response_group.to_dict(),
            "response_scale": self.scale_formula(),
            "predictor_groups": [group.to_dict() for group in self.predictor_groups],
        }


def _as_fraction(value: sp.Rational) -> Fraction:
    rational = sp.Rational(value)
    return Fraction(int(rational.p), int(rational.q))


def _dimension_columns(
    names: Sequence[str], dimensions: Mapping[str, Dimension]
) -> sp.Matrix:
    """Matrix whose columns are the dimension vectors of *names*."""
    return sp.Matrix(
        [
            [sp.Rational(dimensions[name].exponents[row]) for name in names]
            for row in range(len(next(iter(dimensions.values())).exponents))
        ]
    )


def _solve_exponents(
    repeating_matrix: sp.Matrix,
    target_dimension: sp.Matrix,
) -> sp.Matrix | None:
    """Exponents ``a`` with ``repeating_matrix @ a = -target_dimension``.

    Returns ``None`` when the system is inconsistent, meaning the repeating set
    cannot cancel the dimensions of the target variable.
    """
    try:
        solution = repeating_matrix.solve_least_squares(-target_dimension)
    except Exception:
        return None
    if sp.simplify(repeating_matrix * solution + target_dimension) != sp.zeros(
        *target_dimension.shape
    ):
        return None
    return solution


def build_pi_basis(
    response: str,
    feature_units: Mapping[str, str],
    response_unit: str,
    repeating: Sequence[str],
    feature_order: Sequence[str] | None = None,
) -> PiBasis:
    """Construct Pi groups from an explicit repeating-variable set.

    Every non-repeating variable, the response included, yields one group in
    which it appears with exponent one, multiplied by the repeating variables
    raised to whatever exponents cancel its dimensions.

    Raises
    ------
    ValueError
        If the repeating set is dimensionally dependent, or cannot cancel the
        dimensions of some variable.
    """
    names = tuple(feature_order) if feature_order is not None else tuple(feature_units)
    dimensions: dict[str, Dimension] = {
        name: parse_unit(feature_units[name]) for name in names
    }
    dimensions[response] = parse_unit(response_unit)

    repeating = tuple(repeating)
    unknown = [name for name in repeating if name not in dimensions]
    if unknown:
        raise ValueError(f"Unknown repeating variables: {unknown!r}")

    repeating_matrix = _dimension_columns(repeating, dimensions)
    if repeating_matrix.rank() != len(repeating):
        raise ValueError(
            f"Repeating set {repeating!r} is dimensionally dependent "
            f"(rank {repeating_matrix.rank()} < {len(repeating)})."
        )

    def group_for(variable: str, name: str) -> PiGroup:
        target = _dimension_columns([variable], dimensions)
        solution = _solve_exponents(repeating_matrix, target)
        if solution is None:
            raise ValueError(
                f"Repeating set {repeating!r} cannot non-dimensionalise {variable!r}."
            )
        exponents = {variable: Fraction(1)}
        for index, repeater in enumerate(repeating):
            exponents[repeater] = exponents.get(repeater, Fraction(0)) + _as_fraction(
                solution[index]
            )
        return PiGroup(name=name, exponents=exponents)

    response_group = group_for(response, "Pi_0")
    predictors = [name for name in names if name not in repeating]
    predictor_groups = tuple(
        group_for(variable, f"Pi_{index + 1}")
        for index, variable in enumerate(predictors)
    )
    return PiBasis(
        response=response,
        repeating=repeating,
        response_group=response_group,
        predictor_groups=predictor_groups,
    )


@dataclass
class PiBasisScore:
    """How well a candidate basis compresses the response."""

    basis: PiBasis
    log_std_response_group: float
    log_std_response: float

    @property
    def compression(self) -> float:
        """Fraction of log-scale spread removed by non-dimensionalisation."""
        if self.log_std_response <= 0:
            return 0.0
        return 1.0 - self.log_std_response_group / self.log_std_response


def select_pi_basis(
    response: str,
    feature_units: Mapping[str, str],
    response_unit: str,
    columns: Mapping[str, np.ndarray],
    response_values: np.ndarray,
    feature_order: Sequence[str] | None = None,
    max_candidates: int | None = None,
) -> tuple[PiBasis, list[PiBasisScore]]:
    """Choose the repeating set whose response group has the least spread.

    Every admissible repeating set defines a valid and equally rigorous basis.
    Among them, the one whose response group varies least over the design has
    already absorbed the strongest part of the physical scaling, so the relation
    left for the regressor to learn is the simplest.  Ranking bases this way
    makes a step that is normally left to judgement reproducible.

    Returns
    -------
    tuple
        The best basis and the full ranking, best first.
    """
    names = tuple(feature_order) if feature_order is not None else tuple(feature_units)
    dimensions = {name: parse_unit(feature_units[name]) for name in names}
    dimensions[response] = parse_unit(response_unit)

    full_matrix = _dimension_columns(names, dimensions)
    rank = full_matrix.rank()
    if rank == 0:
        raise ValueError("All features are dimensionless; no Pi reduction is possible.")

    response_values = np.asarray(response_values, dtype=np.float64)
    if np.any(response_values <= 0):
        raise ValueError(
            "Pi-basis selection scores the response on a log scale and needs "
            "strictly positive response values."
        )
    log_std_response = float(np.std(np.log(response_values)))

    scores: list[PiBasisScore] = []
    evaluated = 0
    for candidate in itertools.combinations(names, rank):
        if max_candidates is not None and evaluated >= max_candidates:
            break
        try:
            basis = build_pi_basis(
                response, feature_units, response_unit, candidate, feature_order=names
            )
        except ValueError:
            continue
        evaluated += 1

        group_values = basis.to_response_group(columns, response_values)
        if not np.all(np.isfinite(group_values)) or np.any(group_values <= 0):
            continue
        predictors = basis.transform(columns)
        if predictors.size and not np.all(np.isfinite(predictors)):
            continue

        scores.append(
            PiBasisScore(
                basis=basis,
                log_std_response_group=float(np.std(np.log(group_values))),
                log_std_response=log_std_response,
            )
        )

    if not scores:
        raise ValueError("No admissible repeating set produced a usable Pi basis.")

    scores.sort(key=lambda score: score.log_std_response_group)
    best = scores[0]
    logger.info(
        "Pi-basis search: %d admissible bases, best repeating set %s",
        len(scores),
        best.basis.repeating,
    )
    logger.info(
        "  Pi_0 = %s, log-spread %.4f -> %.4f (%.1f%% compression)",
        best.basis.response_group.formula(),
        best.log_std_response,
        best.log_std_response_group,
        100.0 * best.compression,
    )
    for group in best.basis.predictor_groups:
        logger.info("  %s = %s", group.name, group.formula())
    return best.basis, scores


def prune_redundant_groups(
    basis: PiBasis,
    columns: Mapping[str, np.ndarray],
    tolerance: float = 1e-10,
) -> PiBasis:
    """Drop predictor groups that carry no information beyond the others.

    A repeating-variable basis is complete but not always minimal on a given
    design: ratios such as ``depth / l_total`` and ``l_horiz / l_total`` sum to
    one, so one of them is redundant.  A group is dropped when it is constant,
    or when the remaining groups reproduce it by least squares to within
    *tolerance* of unexplained variance.

    Both tests are scale-invariant.  Pi groups are monomials whose magnitude is
    arbitrary -- a group built from a volumetric heat capacity can sit near
    ``1e-30`` while varying over orders of magnitude -- so an absolute variance
    threshold would discard informative groups purely for being small.
    """
    from sklearn.linear_model import LinearRegression

    groups = list(basis.predictor_groups)
    values = [group.evaluate(columns) for group in groups]

    def relative_spread(column: np.ndarray) -> float:
        """Spread measured against the column's own scale."""
        if np.all(column > 0):
            return float(np.std(np.log(column)))
        scale = float(np.mean(np.abs(column)))
        if scale <= 0:
            return 0.0
        return float(np.std(column)) / scale

    keep: list[int] = []
    for index, column in enumerate(values):
        if relative_spread(column) <= tolerance:
            logger.info("Pruning %s: constant over the design.", groups[index].name)
            continue
        # Compare unexplained against explained variance on standardised columns
        # so the ratio does not depend on the group's magnitude.
        centred = (column - column.mean()) / (column.std() or 1.0)
        if keep:
            design = np.column_stack(
                [(values[k] - values[k].mean()) / (values[k].std() or 1.0) for k in keep]
            )
            residual = centred - LinearRegression().fit(design, centred).predict(design)
            if float(np.var(residual)) <= tolerance:
                logger.info(
                    "Pruning %s: reproduced by %s.",
                    groups[index].name,
                    ", ".join(groups[k].name for k in keep),
                )
                continue
        keep.append(index)

    pruned = tuple(
        PiGroup(name=f"Pi_{position + 1}", exponents=groups[index].exponents)
        for position, index in enumerate(keep)
    )
    return PiBasis(
        response=basis.response,
        repeating=basis.repeating,
        response_group=basis.response_group,
        predictor_groups=pruned,
    )


def _rationalise(value: float, max_denominator: int) -> Fraction:
    return Fraction(value).limit_denominator(max_denominator)


def refine_response_group(
    basis: PiBasis,
    columns: Mapping[str, np.ndarray],
    response_values: np.ndarray,
    max_denominator: int = 4,
    rationalise_penalty: float = 0.02,
) -> tuple[PiBasis, dict[str, object]]:
    """Absorb the best power law of the predictor groups into the response group.

    Any product ``Pi_0 * prod(Pi_i ** c_i)`` is dimensionless, so the response
    group produced by a repeating set is only one member of a whole family.  The
    member with the least log-scale spread is available in closed form: the
    exponents that minimise ``var(log Pi_0 + sum c_i log Pi_i)`` are the negated
    coefficients of an ordinary least-squares fit of ``log Pi_0`` on the
    ``log Pi_i``.

    The refined group is the dimensionally consistent power law that best
    explains the response, obtained without any search.  A regressor then only
    has to learn the departure from that power law, which is a markedly easier
    problem than the raw response.

    Exponents are rounded to simple fractions when the extra spread that costs
    stays below *rationalise_penalty* in relative terms, because a formula with
    exponents such as ``3/2`` is far more useful in a paper than one with
    ``1.4971``.
    """
    from sklearn.linear_model import LinearRegression

    if not basis.predictor_groups:
        return basis, {"refined": False, "reason": "no predictor groups"}

    response_values = np.asarray(response_values, dtype=np.float64)
    group_values = basis.to_response_group(columns, response_values)
    all_predictors = basis.transform(columns)
    if np.any(group_values <= 0):
        return basis, {"refined": False, "reason": "non-positive response group"}

    # Only a strictly positive group can be raised to a real power and absorbed;
    # a group built from an already dimensionless, sign-changing variable stays
    # a predictor instead.
    usable = [
        index
        for index in range(all_predictors.shape[1])
        if np.all(all_predictors[:, index] > 0)
    ]
    if not usable:
        return basis, {"refined": False, "reason": "no strictly positive predictor groups"}

    usable_groups = [basis.predictor_groups[index] for index in usable]
    predictors = all_predictors[:, usable]

    log_response = np.log(group_values)
    log_predictors = np.log(predictors)
    fit = LinearRegression().fit(log_predictors, log_response)
    exact_exponents = [-float(coefficient) for coefficient in fit.coef_]

    def spread(exponents: Sequence[float]) -> float:
        adjusted = log_response + log_predictors @ np.asarray(exponents, dtype=np.float64)
        return float(np.std(adjusted))

    baseline = float(np.std(log_response))
    exact_spread = spread(exact_exponents)
    rational_exponents = [_rationalise(value, max_denominator) for value in exact_exponents]
    rational_spread = spread([float(value) for value in rational_exponents])

    use_rational = rational_spread <= exact_spread * (1.0 + rationalise_penalty)
    chosen = (
        [Fraction(value) for value in rational_exponents]
        if use_rational
        else [Fraction(value).limit_denominator(10**6) for value in exact_exponents]
    )
    chosen_spread = rational_spread if use_rational else exact_spread

    combined: dict[str, Fraction] = dict(basis.response_group.exponents)
    for exponent, group in zip(chosen, usable_groups):
        if exponent == 0:
            continue
        for variable, group_exponent in group.exponents.items():
            combined[variable] = combined.get(variable, Fraction(0)) + exponent * group_exponent

    refined_group = PiGroup(
        name="Pi_0", exponents={k: v for k, v in combined.items() if v != 0}
    )
    refined = PiBasis(
        response=basis.response,
        repeating=basis.repeating,
        response_group=refined_group,
        predictor_groups=basis.predictor_groups,
    )

    report = {
        "refined": True,
        "rationalised": use_rational,
        "absorbed_exponents": {
            group.name: str(exponent) for group, exponent in zip(usable_groups, chosen)
        },
        "log_spread_before": baseline,
        "log_spread_after": chosen_spread,
        "power_law": refined_group.formula(),
        "response_scale": refined.scale_formula(),
    }
    logger.info(
        "Response-group refinement: log-spread %.4f -> %.4f; Pi_0 = %s",
        baseline,
        chosen_spread,
        refined_group.formula(),
    )
    return refined, report


def add_physical_constants(
    columns: Mapping[str, np.ndarray],
    units: Mapping[str, str],
    constants: Mapping[str, tuple[float, str]],
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Register fixed physical quantities so they can enter the Pi construction.

    A parameter table often omits a quantity that governs the physics because it
    was held fixed in the study.  Dimensional analysis still needs it: without
    the fluid heat capacity, for instance, a mass flow rate can only be made
    dimensionless through the rock's volumetric heat capacity, which forces an
    inert variable into every flow group.  Declaring the constant restores the
    physically meaningful groups at no cost, since a constant column adds no
    degrees of freedom to the regression.
    """
    length = len(next(iter(columns.values())))
    new_columns = dict(columns)
    new_units = dict(units)
    for name, (value, unit) in constants.items():
        parse_unit(unit)  # Validate eagerly so a typo fails here, not later.
        new_columns[name] = np.full(length, float(value), dtype=np.float64)
        new_units[name] = unit
    return new_columns, new_units


def augment_with_derived_variables(
    columns: Mapping[str, np.ndarray],
    units: Mapping[str, str],
    derived: Mapping[str, tuple[Sequence[str], str]],
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Add summed variables such as a total length to the column set.

    ``derived`` maps a new name to ``(member_names, unit)``.  Members must share
    a dimension; the sum is formed and registered with that unit so it can take
    part in the Pi construction.  Aggregates discovered in target space (see
    :mod:`modeling.structure.discovery`) are fed in here, which lets a variable
    the data itself identified enter the dimensional reduction.
    """
    new_columns = dict(columns)
    new_units = dict(units)
    for name, (members, unit) in derived.items():
        member_dimension = parse_unit(units[members[0]])
        for member in members[1:]:
            if parse_unit(units[member]) != member_dimension:
                raise ValueError(
                    f"Cannot sum {members!r} into {name!r}: dimensions differ."
                )
        total = np.zeros_like(np.asarray(columns[members[0]], dtype=np.float64))
        for member in members:
            total = total + np.asarray(columns[member], dtype=np.float64)
        new_columns[name] = total
        new_units[name] = unit
    return new_columns, new_units
