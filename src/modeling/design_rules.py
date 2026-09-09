"""Analytic design rules extracted from the fitted proxy.

A black-box surrogate answers "what is the cost of this design?".  A closed-form
proxy answers a more useful question: "which design is best, and why?".  Because
every stage of the pipeline is analytic -- a power-law scale, a symbolic residual
and an exact cost identity -- the composed model can be differentiated symbolically
and its stationary points solved for.  The result is a design rule: an explicit
formula for the optimum, valid across the whole parameter space rather than at
the points a numerical optimiser happened to visit.

For the levelised-cost family the structure is
``LCOH(L) = C(L) / w_out(L)`` with a capital cost ``C`` polynomial in the total
drilled length and an output ``w_out`` growing sublinearly with it.  Cost per
unit output therefore falls while the extra length still pays for itself and
rises afterwards, and the stationary point of that trade-off is the economic
optimum.  Solving ``d(log LCOH)/d(log L) = 0`` gives it in closed form.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import sympy as sp

from modeling.structure.identities import TargetIdentity, TargetStructure

logger = logging.getLogger(__name__)


@dataclass
class DesignRule:
    """A stationary point of a target with respect to one design variable."""

    target: str
    variable: str
    condition: str
    solutions: list[str]
    numeric_solutions: list[float] = field(default_factory=list)
    kind: str = "stationary_point"
    notes: str = ""
    design_variable: str | None = None
    output_exponent: float | None = None
    within_sampled_range: bool | None = None
    sampled_range: tuple[float, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "variable": self.variable,
            "design_variable": self.design_variable,
            "output_exponent": self.output_exponent,
            "kind": self.kind,
            "condition": self.condition,
            "solutions": self.solutions,
            "numeric_solutions": self.numeric_solutions,
            "within_sampled_range": self.within_sampled_range,
            "sampled_range": list(self.sampled_range) if self.sampled_range else None,
            "notes": self.notes,
        }


def _cost_polynomial(identity: TargetIdentity) -> tuple[list[float], tuple[str, ...]] | None:
    """Coefficients of a univariate ``product_polynomial`` identity, if it is one."""
    if identity.kind != "product_polynomial":
        return None
    coefficients = identity.parameters.get("coefficients_highest_power_first")
    aggregates = identity.parameters.get("aggregates")
    if coefficients is None or not aggregates or len(aggregates) != 1:
        return None
    return [float(c) for c in coefficients], tuple(aggregates[0])


def optimal_length_rule(
    structure: TargetStructure,
    target: str,
    output_exponent: float,
    variable_name: str = "L",
    evaluate_at: Mapping[str, float] | None = None,
    design_variable: str | None = None,
    sampled_range: tuple[float, float] | None = None,
) -> DesignRule | None:
    """Solve for the drilled length that minimises a levelised cost.

    The proxy gives ``target = C(L) / w_out`` with ``C`` a polynomial in the
    total length ``L``, and the dimensional stage gives ``w_out`` proportional to
    ``L ** output_exponent`` at fixed geology and flow.  Substituting and setting
    the logarithmic derivative to zero,

    ``d/dL [ log C(L) - output_exponent * log L ] = 0``
    ``=>  L * C'(L) - output_exponent * C(L) = 0``

    which is a polynomial equation of the same degree as ``C`` and so is solved
    exactly.  A cost that is quadratic in length gives a single positive root:
    the economic optimum.

    Parameters
    ----------
    structure:
        Fitted target structure holding the cost identity.
    target:
        Name of the levelised-cost target, for example ``"LCOH"``.
    output_exponent:
        Exponent of the length variable in the fitted power law for the output.
        Read off the Pi response scale of the independent target's model.
    variable_name:
        Symbol used for the length in the reported formula.
    evaluate_at:
        Unused placeholder for future context-dependent rules; kept so callers
        can pass a design point without the signature changing.

    Returns
    -------
    DesignRule or None
        ``None`` when the identity for *target* is not a univariate cost
        polynomial, in which case no closed-form rule of this shape exists.
    """
    identity = structure.identities.get(target)
    if identity is None:
        logger.info("No identity for %r; cannot derive a length rule.", target)
        return None
    parsed = _cost_polynomial(identity)
    if parsed is None:
        logger.info("Identity for %r is not a univariate cost polynomial.", target)
        return None

    coefficients, members = parsed
    length = sp.Symbol(variable_name, positive=True)
    degree = len(coefficients) - 1
    cost = sum(
        sp.Float(coefficient) * length ** (degree - power)
        for power, coefficient in enumerate(coefficients)
    )
    exponent = sp.Rational(str(output_exponent)).limit_denominator(1000)

    condition = sp.expand(length * sp.diff(cost, length) - exponent * cost)
    roots = sp.solve(sp.Eq(condition, 0), length)

    positive_roots: list[sp.Expr] = []
    numeric: list[float] = []
    for root in roots:
        value = sp.N(root)
        if value.is_real and float(value) > 0:
            positive_roots.append(sp.nsimplify(root, rational=False))
            numeric.append(float(value))

    extended = f" by extending {design_variable}" if design_variable else ""
    base_notes = (
        f"Derived from {target} = C({variable_name}) / w_out with w_out "
        f"proportional to {variable_name}**{output_exponent} at fixed geology and "
        f"flow rate{extended}. Roots are total drilled length in metres."
    )

    within_range: bool | None = None
    if positive_roots:
        # The second derivative at the root fixes whether the stationary point is
        # a minimum or a maximum of the levelised cost.
        objective = sp.log(cost) - exponent * sp.log(length)
        second = sp.diff(objective, length, 2).subs(length, sp.Float(max(numeric)))
        kind = "interior_minimum" if sp.N(second) > 0 else "interior_maximum"
        notes = base_notes
        if sampled_range is not None:
            low, high = sampled_range
            within_range = any(low <= value <= high for value in numeric)
            if not within_range:
                notes += (
                    f" The root lies outside the sampled range "
                    f"[{low:.0f}, {high:.0f}] m, so it is an extrapolation of the "
                    "fitted cost law rather than an optimum the design actually "
                    "explored; within the sampled range the target is monotone."
                )
        logger.info(
            "Design rule for %s%s: %s at %s = %s m%s",
            target,
            extended,
            kind.replace("_", " "),
            " + ".join(members),
            ", ".join(f"{value:.1f}" for value in numeric),
            "" if within_range is not False else " (outside sampled range)",
        )
    else:
        # No stationary point in the positive domain means the cost is monotone
        # in length, so the optimum sits on the boundary of the design range.
        # Reporting the direction is more useful than reporting no solution.
        probe = sp.diff(sp.log(cost) - exponent * sp.log(length), length)
        slope = sp.N(probe.subs(length, sp.Float(1000.0)))
        decreasing = bool(slope < 0)
        kind = "monotone_decreasing" if decreasing else "monotone_increasing"
        notes = (
            base_notes
            + f" No stationary point exists for {variable_name} > 0: over the "
            f"positive domain {target} is {kind.replace('_', ' ')} in "
            f"{variable_name}, so the optimum lies at the "
            + ("upper" if decreasing else "lower")
            + " end of the drilled-length range rather than inside it. With an "
            f"output exponent of {output_exponent} the extra output from a longer "
            "well always outpaces its cost, which is a statement about the fitted "
            "power law and should be read only inside the sampled range."
        )
        logger.info(
            "Design rule for %s%s: no interior optimum; %s in %s.",
            target,
            extended,
            kind.replace("_", " "),
            " + ".join(members),
        )

    return DesignRule(
        target=target,
        variable=" + ".join(members),
        condition=f"{sp.sstr(condition)} = 0",
        solutions=[sp.sstr(root) for root in positive_roots],
        numeric_solutions=numeric,
        kind=kind,
        notes=notes,
        design_variable=design_variable,
        output_exponent=float(output_exponent),
        within_sampled_range=within_range,
        sampled_range=sampled_range,
    )


def aggregate_key(scale_exponents: Mapping[str, Any], members: Sequence[str]) -> str | None:
    """Name under which the summed length appears in the power law, if it does."""
    candidate = "_plus_".join(members)
    if candidate in scale_exponents:
        return candidate
    for name in scale_exponents:
        if "_plus_" in name and set(name.split("_plus_")) == set(members):
            return name
    return None


def length_exponent_for_variable(
    scale_exponents: Mapping[str, Any],
    members: Sequence[str],
    variable: str,
) -> float:
    """Sensitivity of the output to total length when *variable* is what is extended.

    The response scale is a monomial such as ``k_rock * gradT * l_total * depth``
    in which the total length and the vertical depth appear as separate factors.
    How strongly the output responds to an extra metre of well therefore depends
    on which section is extended, and a single exponent for "length" is
    ambiguous:

    * extending the horizontal section moves ``l_total`` alone, so only the
      exponent on ``l_total`` counts;
    * deepening the well moves ``l_total`` *and* ``depth``, because a deeper well
      is both longer and hotter, so both exponents count.

    Summing every length-bearing factor regardless -- treating the two cases as
    one -- overstates the sensitivity of a horizontal extension and yields a
    design rule for a well that cannot be drilled.  The exponent is therefore
    computed per design variable.
    """
    total = 0.0
    key = aggregate_key(scale_exponents, members)
    if key is not None:
        total += float(sp.Rational(str(scale_exponents[key])))
    if variable in scale_exponents:
        total += float(sp.Rational(str(scale_exponents[variable])))
    return total


def length_exponents_by_variable(
    scale_exponents: Mapping[str, Any],
    members: Sequence[str],
) -> dict[str, float]:
    """One length exponent per section of the well that could be extended."""
    return {
        member: length_exponent_for_variable(scale_exponents, members, member)
        for member in members
    }


def length_exponent_from_scale(
    scale_exponents: Mapping[str, Any],
    length_members: Sequence[str],
) -> float:
    """Largest per-variable length exponent, kept for backward compatibility.

    Prefer :func:`length_exponents_by_variable`, which distinguishes the design
    decisions that this single number conflates.
    """
    exponents = length_exponents_by_variable(scale_exponents, length_members)
    return max(exponents.values()) if exponents else 0.0


def sensitivity_table(
    scale_exponents: Mapping[str, Any],
) -> dict[str, float]:
    """Elasticity of the response with respect to each variable in the power law.

    In a power law the exponent *is* the elasticity: a one percent change in the
    variable moves the response by that many percent.  Reporting the exponents
    directly gives a sensitivity analysis with no sampling and no surrogate of a
    surrogate.
    """
    return {name: float(sp.Rational(str(exponent))) for name, exponent in scale_exponents.items()}
