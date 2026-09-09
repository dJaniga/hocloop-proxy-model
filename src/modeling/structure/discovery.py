"""Automatic discovery of exact algebraic structure in the target space.

Multi-output simulator studies frequently report target columns that are not
mutually independent: one output may be a fixed multiple of another, or the
product of two outputs may collapse onto a low-order polynomial of an
aggregate of the input parameters.  Fitting an independent surrogate per target
then wastes model capacity *and* allows the surrogate to produce combinations of
outputs the simulator could never emit.

This module searches for such relations before any model is trained.  Every
candidate is accepted only when it reproduces the target to within a strict
relative-error tolerance, so an accepted identity is exact up to the numerical
precision of the source data.  The surviving *independent* targets are the only
ones a regression model has to learn; the rest are reconstructed in closed form,
which makes the proxy consistent with the discovered relations by construction.

The search covers four functional families:

``proportional``
    ``y_i = k * y_j``
``affine``
    ``y_i = a * y_j + b``
``product_polynomial``
    ``y_i * y_j = P(v)`` or ``y_i / y_j = P(v)`` for a polynomial ``P`` in
    aggregates ``v`` of the features
``feature_polynomial``
    ``y_i = P(v)``, i.e. the target is a closed-form function of the inputs alone

Feature aggregates are themselves discovered: an ordinary least-squares fit of
the derived quantity on the raw features is inspected for coefficients that
coincide within a relative tolerance, and the corresponding features are summed
into a single variable.  This is what recovers a total-length variable such as
``depth + l_horiz`` without it being supplied by the user.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
from numpy.polynomial import Polynomial
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from modeling.structure.identities import (
    IdentityQuality,
    TargetIdentity,
    TargetStructure,
    format_polynomial,
    relative_error_quality,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Configuration                                                                #
# --------------------------------------------------------------------------- #


@dataclass
class DiscoveryConfig:
    """Tuning knobs for :func:`discover_target_structure`."""

    #: Maximum relative error a reconstruction may show at any sample and still
    #: be accepted as exact.  1e-6 is loose enough for data stored with ~10
    #: significant digits and tight enough to reject a merely good fit.
    tolerance: float = 1e-6
    #: Highest polynomial degree tried when fitting a derived quantity.
    max_polynomial_degree: int = 3
    #: Relative tolerances tried when declaring two OLS coefficients equal, which
    #: is the test for grouping features into a summed aggregate.  A ladder is
    #: used rather than one value because the right threshold depends on how
    #: precisely the coefficients are estimated, which varies with sample size.
    #: Every grouping is validated against :attr:`tolerance` afterwards, so an
    #: over-eager grouping is rejected rather than believed.
    coefficient_match_rtols: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 5e-2)
    #: Thresholds tried for dropping a feature whose standardised contribution to
    #: the derived quantity is negligible.
    negligible_contributions: tuple[float, ...] = (1e-3, 1e-2, 5e-2)
    #: Folds used by the learnability score that picks which targets to learn.
    learnability_folds: int = 3
    #: Samples required per fitted polynomial coefficient.  A polynomial with as
    #: many coefficients as there are samples reproduces anything exactly, so a
    #: fit with too little data per parameter is not evidence of an identity.
    min_samples_per_parameter: int = 10
    #: Fraction of the data withheld when confirming a candidate identity.  An
    #: exact algebraic relation holds on data it was not fitted to; an
    #: overparameterised fit does not, which is what this test separates.
    validation_fraction: float = 0.3
    #: Relative tolerance allowed on the withheld data.  Looser than
    #: :attr:`tolerance` because the coefficients come from a smaller fit and
    #: carry correspondingly more rounding error.
    validation_tolerance: float = 1e-4
    #: Two candidate independent sets whose reconstruction scores differ by less
    #: than this are treated as tied, and the tie is broken by learnability.
    selection_tolerance: float = 0.05
    #: Random seed for the learnability score and the validation split.
    seed: int | None = 0


# --------------------------------------------------------------------------- #
# Feature aggregation                                                          #
# --------------------------------------------------------------------------- #


@dataclass
class FeatureAggregate:
    """A summed group of features treated as one variable."""

    name: str
    members: tuple[str, ...]

    def values(self, feature_values: dict[str, np.ndarray]) -> np.ndarray:
        total = np.zeros_like(np.asarray(feature_values[self.members[0]], dtype=np.float64))
        for member in self.members:
            total = total + np.asarray(feature_values[member], dtype=np.float64)
        return total


def _find_feature_aggregates(
    features: np.ndarray,
    feature_names: Sequence[str],
    derived: np.ndarray,
    coefficient_match_rtol: float,
    negligible_contribution: float,
    unit_groups: dict[str, str] | None = None,
) -> list[FeatureAggregate]:
    """Group features that enter *derived* with an identical linear coefficient.

    Features are first screened for a non-negligible standardised contribution.
    Surviving features whose raw OLS coefficients agree within
    *coefficient_match_rtol* are summed into one aggregate, because an equal
    coefficient means the fit only ever sees their sum.  When ``unit_groups`` is
    supplied, only features sharing a physical dimension may be grouped, which
    keeps the aggregate physically meaningful.
    """
    model = LinearRegression().fit(features, derived)
    coefficients = np.asarray(model.coef_, dtype=np.float64).ravel()

    derived_scale = float(np.std(derived))
    if derived_scale <= 0:
        derived_scale = 1.0
    contributions = np.abs(coefficients) * features.std(axis=0) / derived_scale
    relevant = [
        index
        for index in range(len(feature_names))
        if contributions[index] >= negligible_contribution
    ]

    aggregates: list[FeatureAggregate] = []
    unassigned = list(relevant)
    while unassigned:
        pivot = unassigned.pop(0)
        group = [pivot]
        for other in list(unassigned):
            same_unit = (
                unit_groups is None
                or unit_groups.get(feature_names[pivot]) == unit_groups.get(feature_names[other])
            )
            if not same_unit:
                continue
            reference = max(abs(coefficients[pivot]), abs(coefficients[other]))
            if reference == 0:
                continue
            if abs(coefficients[pivot] - coefficients[other]) / reference <= coefficient_match_rtol:
                group.append(other)
                unassigned.remove(other)
        members = tuple(feature_names[index] for index in group)
        name = members[0] if len(members) == 1 else "(" + " + ".join(members) + ")"
        aggregates.append(FeatureAggregate(name=name, members=members))

    return aggregates


def _aggregate_candidates(
    features: np.ndarray,
    feature_names: Sequence[str],
    derived: np.ndarray,
    config: DiscoveryConfig,
    unit_groups: dict[str, str] | None,
) -> list[list[FeatureAggregate]]:
    """Distinct groupings to try, simplest first.

    Whether two estimated coefficients count as equal depends on how precisely
    they are estimated, so no single threshold is right for every data set.  The
    search therefore walks a ladder of thresholds and collects the distinct
    groupings they produce.  Fewer aggregates means a simpler identity, so those
    are tried first; correctness is not at stake because every candidate must
    still reproduce the target to within :attr:`DiscoveryConfig.tolerance`.
    """
    seen: set[tuple[tuple[str, ...], ...]] = set()
    candidates: list[list[FeatureAggregate]] = []
    for rtol in config.coefficient_match_rtols:
        for threshold in config.negligible_contributions:
            aggregates = _find_feature_aggregates(
                features, feature_names, derived, rtol, threshold, unit_groups
            )
            if not aggregates:
                continue
            key = tuple(sorted(aggregate.members for aggregate in aggregates))
            if key in seen:
                continue
            seen.add(key)
            candidates.append(aggregates)
    candidates.sort(key=len)
    return candidates


# --------------------------------------------------------------------------- #
# Polynomial fitting on aggregates                                             #
# --------------------------------------------------------------------------- #


@dataclass
class PolynomialModel:
    """A fitted polynomial in one or more feature aggregates."""

    aggregates: tuple[FeatureAggregate, ...]
    degree: int
    expression: str
    parameters: dict[str, Any]
    _predict: Any

    def predict(self, feature_values: dict[str, np.ndarray]) -> np.ndarray:
        columns = [aggregate.values(feature_values) for aggregate in self.aggregates]
        return self._predict(np.column_stack(columns))

    @property
    def feature_names(self) -> tuple[str, ...]:
        names: list[str] = []
        for aggregate in self.aggregates:
            names.extend(aggregate.members)
        return tuple(dict.fromkeys(names))


def _fit_polynomial(
    aggregates: Sequence[FeatureAggregate],
    aggregate_matrix: np.ndarray,
    derived: np.ndarray,
    degree: int,
) -> PolynomialModel:
    """Fit a total-degree polynomial of *derived* on the aggregate columns."""
    if len(aggregates) == 1:
        # A univariate fit on a scaled domain is numerically far better behaved
        # than a raw Vandermonde solve when the variable spans thousands of units.
        series = Polynomial.fit(aggregate_matrix[:, 0], derived, deg=degree)
        coefficients_ascending = series.convert().coef
        coefficients = list(coefficients_ascending[::-1])

        def predict(matrix: np.ndarray, _series=series) -> np.ndarray:
            return _series(matrix[:, 0])

        return PolynomialModel(
            aggregates=tuple(aggregates),
            degree=degree,
            expression=format_polynomial(coefficients, aggregates[0].name),
            parameters={
                "coefficients_highest_power_first": [float(c) for c in coefficients],
                "variable": aggregates[0].name,
            },
            _predict=predict,
        )

    pipeline = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=degree, include_bias=False),
        LinearRegression(),
    ).fit(aggregate_matrix, derived)
    names = [aggregate.name for aggregate in aggregates]
    expression = f"polynomial(degree={degree}) in {', '.join(names)}"
    return PolynomialModel(
        aggregates=tuple(aggregates),
        degree=degree,
        expression=expression,
        parameters={"variables": names, "degree": degree},
        _predict=pipeline.predict,
    )


def _parameter_count(n_variables: int, degree: int) -> int:
    """Number of coefficients in a total-degree polynomial, including the constant."""
    from math import comb

    return comb(n_variables + degree, degree)


def _fit_on_subset(
    aggregates: Sequence[FeatureAggregate],
    feature_values: dict[str, np.ndarray],
    derived: np.ndarray,
    config: DiscoveryConfig,
) -> tuple[PolynomialModel, IdentityQuality] | None:
    """Lowest-degree polynomial in *aggregates* that reproduces *derived*."""
    if not aggregates:
        return None
    aggregate_matrix = np.column_stack(
        [aggregate.values(feature_values) for aggregate in aggregates]
    )
    n_samples = derived.shape[0]
    best: tuple[PolynomialModel, IdentityQuality] | None = None
    for degree in range(1, config.max_polynomial_degree + 1):
        parameters = _parameter_count(len(aggregates), degree)
        if n_samples < config.min_samples_per_parameter * parameters:
            # Too little data to distinguish an identity from a flexible fit.
            break
        model = _fit_polynomial(aggregates, aggregate_matrix, derived, degree)
        quality = relative_error_quality(derived, model.predict(feature_values))
        if best is None or quality.max_abs_relative_error < best[1].max_abs_relative_error:
            best = (model, quality)
        if quality.is_exact(config.tolerance):
            # Lowest exact degree wins; a higher degree would only overfit.
            return model, quality
    return best


def _holds_out_of_sample(
    aggregates: Sequence[FeatureAggregate],
    feature_values: dict[str, np.ndarray],
    derived: np.ndarray,
    degree: int,
    config: DiscoveryConfig,
) -> bool:
    """Refit the polynomial on part of the data and test it on the rest.

    An exact algebraic relation is a property of the system, so coefficients
    estimated from a subset reproduce the whole.  A fit that is exact only
    because it has enough free parameters to interpolate the sample fails here.
    This is the test that stops a spurious identity being accepted on a small
    design, where an in-sample residual of zero carries almost no information.
    """
    n_samples = derived.shape[0]
    n_fit = int(round(n_samples * (1.0 - config.validation_fraction)))
    if n_fit < 2 or n_fit >= n_samples:
        return True  # Too small to split; the parameter-count guard applies instead.

    rng = np.random.default_rng(config.seed)
    order = rng.permutation(n_samples)
    fit_index, test_index = order[:n_fit], order[n_fit:]

    fit_values = {name: column[fit_index] for name, column in feature_values.items()}
    test_values = {name: column[test_index] for name, column in feature_values.items()}
    fit_matrix = np.column_stack(
        [aggregate.values(fit_values) for aggregate in aggregates]
    )
    try:
        model = _fit_polynomial(aggregates, fit_matrix, derived[fit_index], degree)
        quality = relative_error_quality(derived[test_index], model.predict(test_values))
    except Exception:
        return False
    return quality.max_abs_relative_error <= config.validation_tolerance


def _minimise_variables(
    aggregates: Sequence[FeatureAggregate],
    feature_values: dict[str, np.ndarray],
    derived: np.ndarray,
    config: DiscoveryConfig,
) -> tuple[PolynomialModel, IdentityQuality] | None:
    """Drop variables greedily for as long as the fit stays exact.

    A polynomial of degree two in four variables has enough freedom to absorb a
    relation that genuinely involves only one, so an exact fit on the full
    variable set is not evidence that every variable belongs in the identity.
    What survives elimination is the minimal exact identity, which is the one
    worth reporting.
    """
    found = _fit_on_subset(aggregates, feature_values, derived, config)
    if found is None or not found[1].is_exact(config.tolerance):
        return found

    current = list(aggregates)
    best = found
    improved = True
    while improved and len(current) > 1:
        improved = False
        for candidate in list(current):
            reduced = [item for item in current if item is not candidate]
            trial = _fit_on_subset(reduced, feature_values, derived, config)
            if trial is not None and trial[1].is_exact(config.tolerance):
                logger.debug("Dropping %s from identity; fit stays exact.", candidate.name)
                current = reduced
                best = trial
                improved = True
                break
    return best


def _best_polynomial(
    features: np.ndarray,
    feature_names: Sequence[str],
    feature_values: dict[str, np.ndarray],
    derived: np.ndarray,
    config: DiscoveryConfig,
    unit_groups: dict[str, str] | None,
) -> tuple[PolynomialModel, IdentityQuality] | None:
    """Simplest exact polynomial reproduction of *derived*, over all groupings.

    Each candidate grouping is fitted and then stripped of superfluous
    variables; the first grouping that yields an exact fit wins, and because
    candidates arrive simplest-first that is also the simplest exact identity.
    When none is exact the closest attempt is returned so the caller can report
    how near the data came to a relation.
    """
    candidates = _aggregate_candidates(
        features, feature_names, derived, config, unit_groups
    )
    closest: tuple[PolynomialModel, IdentityQuality] | None = None
    for aggregates in candidates:
        found = _minimise_variables(aggregates, feature_values, derived, config)
        if found is None:
            continue
        if found[1].is_exact(config.tolerance):
            model, _ = found
            if not _holds_out_of_sample(
                model.aggregates, feature_values, derived, model.degree, config
            ):
                logger.debug(
                    "Rejecting candidate identity in %s: exact in sample but not "
                    "on withheld data.",
                    [aggregate.name for aggregate in model.aggregates],
                )
                continue
            return found
        if closest is None or found[1].max_abs_relative_error < closest[1].max_abs_relative_error:
            closest = found
    return closest


# --------------------------------------------------------------------------- #
# Candidate identity search                                                    #
# --------------------------------------------------------------------------- #


def _proportional_candidate(
    target: str,
    source: str,
    y_target: np.ndarray,
    y_source: np.ndarray,
    config: DiscoveryConfig,
) -> TargetIdentity | None:
    if np.any(y_source == 0):
        return None
    ratio = y_target / y_source
    factor = float(np.mean(ratio))
    quality = relative_error_quality(y_target, factor * y_source)
    if not quality.is_exact(config.tolerance):
        return None

    def evaluate(targets: dict[str, np.ndarray], _features: dict[str, np.ndarray],
                 _source=source, _factor=factor) -> np.ndarray:
        return _factor * np.asarray(targets[_source], dtype=np.float64)

    return TargetIdentity(
        target=target,
        source_targets=(source,),
        source_features=(),
        kind="proportional",
        expression=f"{target} = {factor:.12g} * {source}",
        quality=quality,
        parameters={"factor": factor},
        evaluate=evaluate,
    )


def _affine_candidate(
    target: str,
    source: str,
    y_target: np.ndarray,
    y_source: np.ndarray,
    config: DiscoveryConfig,
) -> TargetIdentity | None:
    model = LinearRegression().fit(y_source.reshape(-1, 1), y_target)
    slope = float(model.coef_[0])
    intercept = float(model.intercept_)
    quality = relative_error_quality(y_target, slope * y_source + intercept)
    if not quality.is_exact(config.tolerance):
        return None

    def evaluate(targets: dict[str, np.ndarray], _features: dict[str, np.ndarray],
                 _source=source, _slope=slope, _intercept=intercept) -> np.ndarray:
        return _slope * np.asarray(targets[_source], dtype=np.float64) + _intercept

    return TargetIdentity(
        target=target,
        source_targets=(source,),
        source_features=(),
        kind="affine",
        expression=f"{target} = {slope:.12g} * {source} + {intercept:.12g}",
        quality=quality,
        parameters={"slope": slope, "intercept": intercept},
        evaluate=evaluate,
    )


def _product_polynomial_candidate(
    target: str,
    source: str,
    y_target: np.ndarray,
    y_source: np.ndarray,
    features: np.ndarray,
    feature_names: Sequence[str],
    feature_values: dict[str, np.ndarray],
    config: DiscoveryConfig,
    unit_groups: dict[str, str] | None,
    combination: str,
) -> TargetIdentity | None:
    """Test whether ``target * source`` or ``target / source`` is a polynomial.

    A hit means the target is reconstructed as ``P(v) / source`` respectively
    ``P(v) * source``, which keeps the pair exactly consistent by construction.
    """
    if combination == "product":
        derived = y_target * y_source
    elif combination == "quotient":
        if np.any(y_source == 0):
            return None
        derived = y_target / y_source
    else:
        raise ValueError(f"Unknown combination {combination!r}")

    if not np.all(np.isfinite(derived)):
        return None

    found = _best_polynomial(
        features, feature_names, feature_values, derived, config, unit_groups
    )
    if found is None:
        return None
    polynomial, derived_quality = found
    if not derived_quality.is_exact(config.tolerance):
        return None

    if combination == "product":
        symbol, inverse = "*", "/"
    else:
        symbol, inverse = "/", "*"

    def evaluate(targets: dict[str, np.ndarray], feats: dict[str, np.ndarray],
                 _source=source, _poly=polynomial, _combination=combination) -> np.ndarray:
        base = _poly.predict(feats)
        source_values = np.asarray(targets[_source], dtype=np.float64)
        if _combination == "product":
            return base / source_values
        return base * source_values

    reconstructed = evaluate(
        {source: y_source}, feature_values
    )
    quality = relative_error_quality(y_target, reconstructed)
    if not quality.is_exact(config.tolerance):
        return None

    return TargetIdentity(
        target=target,
        source_targets=(source,),
        source_features=polynomial.feature_names,
        kind=f"{combination}_polynomial",
        expression=(
            f"{target} {symbol} {source} = {polynomial.expression}"
            f"   =>   {target} = ({polynomial.expression}) {inverse} {source}"
        ),
        quality=quality,
        parameters={
            "combination": combination,
            "degree": polynomial.degree,
            "aggregates": [list(a.members) for a in polynomial.aggregates],
            **polynomial.parameters,
        },
        evaluate=evaluate,
    )


def _feature_polynomial_candidate(
    target: str,
    y_target: np.ndarray,
    features: np.ndarray,
    feature_names: Sequence[str],
    feature_values: dict[str, np.ndarray],
    config: DiscoveryConfig,
    unit_groups: dict[str, str] | None,
) -> TargetIdentity | None:
    """Test whether the target is exactly a polynomial of the features alone."""
    found = _best_polynomial(
        features, feature_names, feature_values, y_target, config, unit_groups
    )
    if found is None:
        return None
    polynomial, quality = found
    if not quality.is_exact(config.tolerance):
        return None

    def evaluate(_targets: dict[str, np.ndarray], feats: dict[str, np.ndarray],
                 _poly=polynomial) -> np.ndarray:
        return _poly.predict(feats)

    return TargetIdentity(
        target=target,
        source_targets=(),
        source_features=polynomial.feature_names,
        kind="feature_polynomial",
        expression=f"{target} = {polynomial.expression}",
        quality=quality,
        parameters={
            "degree": polynomial.degree,
            "aggregates": [list(a.members) for a in polynomial.aggregates],
            **polynomial.parameters,
        },
        evaluate=evaluate,
    )


# --------------------------------------------------------------------------- #
# Learnability score and independent-set selection                             #
# --------------------------------------------------------------------------- #


def _learnability(
    features: np.ndarray,
    y: np.ndarray,
    config: DiscoveryConfig,
) -> float:
    """Cheap cross-validated R-squared used to rank candidate learning targets.

    The score is computed on ``log(y)`` when the target is strictly positive,
    because the surrogate downstream is fitted in log space for such targets.
    """
    response = np.log(y) if np.all(y > 0) else y
    model = HistGradientBoostingRegressor(max_iter=150, random_state=config.seed)
    folds = KFold(n_splits=config.learnability_folds, shuffle=True, random_state=config.seed)
    try:
        return float(np.mean(cross_val_score(model, features, response, cv=folds, scoring="r2")))
    except Exception:  # pragma: no cover - defensive, scoring must never abort discovery
        logger.warning("Learnability score failed; treating target as hard to learn.")
        return -np.inf


def _closure(
    seeds: Iterable[str],
    candidates: dict[str, list[TargetIdentity]],
) -> set[str]:
    """All targets reachable from *seeds* by repeatedly applying identities."""
    known = set(seeds)
    changed = True
    while changed:
        changed = False
        for target, identity_list in candidates.items():
            if target in known:
                continue
            for identity in identity_list:
                if all(source in known for source in identity.source_targets):
                    known.add(target)
                    changed = True
                    break
    return known


def _assemble_identities(
    target_names: Sequence[str],
    independent: Sequence[str],
    candidates: dict[str, list[TargetIdentity]],
) -> dict[str, TargetIdentity]:
    """Pick one identity per derived target, resolvable from *independent*.

    Preference goes to the identity with the fewest source targets and then the
    smallest residual, so a reconstruction chain stays as short as possible.
    """
    identities: dict[str, TargetIdentity] = {}
    resolved = set(independent)
    pending = [name for name in target_names if name not in resolved]
    while pending:
        progressed = False
        for name in list(pending):
            usable = [
                identity
                for identity in candidates[name]
                if all(source in resolved for source in identity.source_targets)
            ]
            if not usable:
                continue
            identities[name] = min(
                usable,
                key=lambda i: (len(i.source_targets), i.quality.max_abs_relative_error),
            )
            resolved.add(name)
            pending.remove(name)
            progressed = True
        if not progressed:
            break
    return identities


def _reconstruction_score(
    features: np.ndarray,
    target_values: dict[str, np.ndarray],
    feature_values: dict[str, np.ndarray],
    target_names: Sequence[str],
    independent: Sequence[str],
    candidates: dict[str, list[TargetIdentity]],
    config: DiscoveryConfig,
) -> float:
    """Score a candidate independent set by how well *every* target ends up predicted.

    Which target to learn cannot be decided by looking at that target alone.  The
    identities reconstruct the others by division, and division preserves
    relative error while amplifying absolute error wherever the divisor is small.
    Learning the wrong member of the set therefore produces a model that is
    accurate in relative terms and yet has a hugely negative R-squared on a
    target spanning orders of magnitude.

    Each candidate set is scored on what actually matters: out-of-fold
    predictions of its independent targets are pushed through the identities and
    the reconstructed values of *all* targets are scored on the raw scale, which
    is where the amplification shows up.

    The score is the *worst* target's R-squared, not the average.  Averaging
    rewards a set for covering several targets directly -- and when two targets
    are proportional to one another, learning either scores both -- so it can
    prefer a set whose one reconstructed target is catastrophic.  Taking the
    minimum states the real requirement: no target may collapse.
    """
    identities = _assemble_identities(target_names, independent, candidates)
    if len(identities) + len(independent) < len(target_names):
        return -np.inf

    folds = KFold(n_splits=config.learnability_folds, shuffle=True, random_state=config.seed)
    predictions: dict[str, np.ndarray] = {}
    for name in independent:
        y = target_values[name]
        positive = bool(np.all(y > 0))
        response = np.log(y) if positive else y
        model = HistGradientBoostingRegressor(max_iter=150, random_state=config.seed)
        try:
            out_of_fold = cross_val_predict(model, features, response, cv=folds)
        except Exception:  # pragma: no cover - scoring must never abort discovery
            return -np.inf
        predictions[name] = np.exp(out_of_fold) if positive else out_of_fold

    known = dict(predictions)
    for name in target_names:
        if name in known or name not in identities:
            continue
    # Resolve in dependency order, exactly as the fitted model will.
    pending = [name for name in target_names if name not in known]
    while pending:
        progressed = False
        for name in list(pending):
            identity = identities[name]
            if all(source in known for source in identity.source_targets):
                with np.errstate(divide="ignore", invalid="ignore"):
                    known[name] = identity(known, feature_values)
                pending.remove(name)
                progressed = True
        if not progressed:
            return -np.inf

    scores: list[float] = []
    for name in target_names:
        estimate = np.asarray(known[name], dtype=np.float64)
        actual = target_values[name]
        if not np.all(np.isfinite(estimate)):
            return -np.inf
        denominator = float(np.sum((actual - actual.mean()) ** 2))
        if denominator <= 0:
            continue
        scores.append(1.0 - float(np.sum((actual - estimate) ** 2)) / denominator)
    return float(np.min(scores)) if scores else -np.inf


def _select_independent_targets(
    features: np.ndarray,
    target_values: dict[str, np.ndarray],
    feature_values: dict[str, np.ndarray],
    target_names: Sequence[str],
    candidates: dict[str, list[TargetIdentity]],
    config: DiscoveryConfig,
) -> tuple[str, ...]:
    """Smallest set of targets that reconstructs all others, best set first.

    Subsets are enumerated by increasing size, so the first size at which a
    covering set exists is minimal.  Choosing among them takes two criteria in
    order, because either alone picks badly:

    *Reconstruction safety first.*  :func:`_reconstruction_score` rejects a set
    whose identities would amplify error catastrophically on some target.

    *Then learnability.*  When several sets are within
    ``selection_tolerance`` of the best reconstruction score, the difference
    between them is noise, and picking on noise has a real cost: the choice
    determines which quantity the dimensional stage builds its power law for.
    Learning the primary physical output gives an interpretable law with small
    integer exponents, while learning a quantity derived from it gives the same
    accuracy expressed in sixteenths.  The near-tie is therefore broken by
    log-scale learnability, which prefers the target a model handles most
    directly.
    """
    all_targets = set(target_names)
    for size in range(1, len(target_names) + 1):
        covering = [
            subset
            for subset in itertools.combinations(target_names, size)
            if _closure(subset, candidates) >= all_targets
        ]
        if not covering:
            continue
        scored = {
            subset: _reconstruction_score(
                features, target_values, feature_values, target_names,
                subset, candidates, config,
            )
            for subset in covering
        }
        logger.info(
            "Reconstruction scores: %s",
            {", ".join(k): round(v, 4) for k, v in scored.items()},
        )

        best_score = max(scored.values())
        shortlist = [
            subset
            for subset, score in scored.items()
            if score >= best_score - config.selection_tolerance
        ]
        if len(shortlist) == 1:
            return shortlist[0]

        learnability = {
            name: _learnability(features, target_values[name], config)
            for name in {name for subset in shortlist for name in subset}
        }
        logger.info(
            "Near-tie among %d sets; learnability: %s",
            len(shortlist),
            {k: round(v, 4) for k, v in learnability.items()},
        )
        return max(
            shortlist,
            key=lambda subset: sum(learnability.get(name, -np.inf) for name in subset),
        )
    return tuple(target_names)


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #


def discover_target_structure(
    features: np.ndarray,
    targets: np.ndarray,
    feature_names: Sequence[str],
    target_names: Sequence[str],
    config: DiscoveryConfig | None = None,
    unit_groups: dict[str, str] | None = None,
) -> TargetStructure:
    """Search the target space for exact closed-form relations.

    Parameters
    ----------
    features:
        Feature matrix of shape ``(n_samples, n_features)``.
    targets:
        Target matrix of shape ``(n_samples, n_targets)``.
    feature_names, target_names:
        Column names, used for the reported formulas.
    config:
        Search settings; see :class:`DiscoveryConfig`.
    unit_groups:
        Optional mapping from feature name to a physical-dimension label.  When
        given, only features of the same dimension may be summed into an
        aggregate, which rules out dimensionally meaningless groupings.

    Returns
    -------
    TargetStructure
        The independent targets to learn plus the accepted identities.
    """
    config = config or DiscoveryConfig()
    features = np.asarray(features, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if targets.ndim == 1:
        targets = targets.reshape(-1, 1)

    feature_names = tuple(feature_names)
    target_names = tuple(target_names)
    feature_values = {name: features[:, i] for i, name in enumerate(feature_names)}
    target_values = {name: targets[:, k] for k, name in enumerate(target_names)}

    logger.info(
        "Target-structure discovery over %d targets and %d features",
        len(target_names),
        len(feature_names),
    )

    candidates: dict[str, list[TargetIdentity]] = {name: [] for name in target_names}
    rejected: list[dict[str, Any]] = []

    for target in target_names:
        y_target = target_values[target]

        standalone = _feature_polynomial_candidate(
            target, y_target, features, feature_names, feature_values, config, unit_groups
        )
        if standalone is not None:
            candidates[target].append(standalone)

        for source in target_names:
            if source == target:
                continue
            y_source = target_values[source]

            found = _proportional_candidate(target, source, y_target, y_source, config)
            if found is None:
                found = _affine_candidate(target, source, y_target, y_source, config)
            if found is None:
                for combination in ("product", "quotient"):
                    found = _product_polynomial_candidate(
                        target, source, y_target, y_source, features, feature_names,
                        feature_values, config, unit_groups, combination,
                    )
                    if found is not None:
                        break
            if found is not None:
                candidates[target].append(found)
                logger.info("Identity accepted: %s", found.expression)
            else:
                rejected.append({"target": target, "source": source})

    independent = _select_independent_targets(
        features, target_values, feature_values, target_names, candidates, config
    )

    identities = _assemble_identities(target_names, independent, candidates)
    undeliverable = [
        name
        for name in target_names
        if name not in identities and name not in independent
    ]
    if undeliverable:
        # Anything the identities cannot reach has to be learnt directly.
        independent = independent + tuple(undeliverable)
        identities = _assemble_identities(target_names, independent, candidates)

    structure = TargetStructure(
        target_names=target_names,
        independent_targets=tuple(independent),
        identities=identities,
        rejected=rejected,
        tolerance=config.tolerance,
    )
    logger.info(
        "Structure discovery reduced %d targets to %d independent (%s)",
        len(target_names),
        len(structure.independent_targets),
        ", ".join(structure.independent_targets),
    )
    return structure
