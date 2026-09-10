"""Minimal SI dimensional algebra, with scale factors and currency.

Two things are tracked for every unit: its *dimension*, a vector of base-unit
exponents, and its *scale* relative to the coherent SI unit of that dimension.
Dimension is what the Buckingham-Pi construction needs; scale is what makes
``kW`` different from ``W`` and ``cE/kWh`` different from ``E/kWh``.

Scale never changes a fitted model. A dimensionless group stays dimensionless
whichever multiple of a unit its variables are expressed in, and a constant
factor in the response scale is absorbed by the fitted coefficient. Scale is
tracked because it is needed to *report* a formula in the units the data is
tabulated in, and to catch the one case where ignoring it is silently wrong:
summing two columns of the same dimension but different scale, such as a depth
in metres and a length in kilometres.

Currency is carried as its own base dimension, so a levelised cost in ``E/kWh``
is dimensionally distinct from a pure number. A parameter table with no other
monetary quantity then has no way to make such a target dimensionless, and the
Pi stage reports that and steps aside rather than inventing a group.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Mapping

#: Base dimensions, in the order used by :attr:`Dimension.exponents`.
#: ``Currency`` is not an SI base dimension but behaves like one: it cannot be
#: derived from the others, which is exactly the property the Pi construction
#: relies on.
BASE_DIMENSIONS: tuple[str, ...] = ("M", "L", "T", "Theta", "I", "N", "J", "Currency")

_ZERO = (Fraction(0),) * len(BASE_DIMENSIONS)


@dataclass(frozen=True)
class Dimension:
    """A physical dimension as a vector of base-unit exponents."""

    exponents: tuple[Fraction, ...] = _ZERO

    def __post_init__(self) -> None:
        if len(self.exponents) != len(BASE_DIMENSIONS):
            raise ValueError(
                f"Expected {len(BASE_DIMENSIONS)} exponents, got {len(self.exponents)}."
            )

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, float | int | Fraction]) -> Dimension:
        exponents = []
        for name in BASE_DIMENSIONS:
            exponents.append(Fraction(mapping.get(name, 0)).limit_denominator(1000))
        return cls(tuple(exponents))

    @property
    def is_dimensionless(self) -> bool:
        return all(exponent == 0 for exponent in self.exponents)

    def __mul__(self, other: Dimension) -> Dimension:
        return Dimension(tuple(a + b for a, b in zip(self.exponents, other.exponents)))

    def __truediv__(self, other: Dimension) -> Dimension:
        return Dimension(tuple(a - b for a, b in zip(self.exponents, other.exponents)))

    def __pow__(self, power: int | Fraction) -> Dimension:
        factor = Fraction(power)
        return Dimension(tuple(exponent * factor for exponent in self.exponents))

    def __str__(self) -> str:
        if self.is_dimensionless:
            return "1"
        parts = []
        for name, exponent in zip(BASE_DIMENSIONS, self.exponents):
            if exponent == 0:
                continue
            parts.append(name if exponent == 1 else f"{name}^{exponent}")
        return "*".join(parts)


DIMENSIONLESS = Dimension()


@dataclass(frozen=True)
class Quantity:
    """A unit: a dimension together with its size relative to coherent SI."""

    dimension: Dimension
    scale: float = 1.0

    def __mul__(self, other: Quantity) -> Quantity:
        return Quantity(self.dimension * other.dimension, self.scale * other.scale)

    def __truediv__(self, other: Quantity) -> Quantity:
        return Quantity(self.dimension / other.dimension, self.scale / other.scale)

    def __pow__(self, power: int | Fraction) -> Quantity:
        return Quantity(self.dimension ** power, self.scale ** float(power))


#: Unit symbols understood by :func:`parse_unit`, before prefixes are applied.
#: An exact match here always wins over a prefix interpretation, which is what
#: keeps ``m`` a metre rather than a milli-something and ``cd`` a candela rather
#: than a centi-day.
UNIT_SYMBOLS: dict[str, Quantity] = {
    "1": Quantity(DIMENSIONLESS),
    "-": Quantity(DIMENSIONLESS),
    "%": Quantity(DIMENSIONLESS, 0.01),
    "kg": Quantity(Dimension.from_mapping({"M": 1})),
    "g": Quantity(Dimension.from_mapping({"M": 1}), 1e-3),
    "m": Quantity(Dimension.from_mapping({"L": 1})),
    "s": Quantity(Dimension.from_mapping({"T": 1})),
    "h": Quantity(Dimension.from_mapping({"T": 1}), 3600.0),
    "K": Quantity(Dimension.from_mapping({"Theta": 1})),
    "A": Quantity(Dimension.from_mapping({"I": 1})),
    "mol": Quantity(Dimension.from_mapping({"N": 1})),
    "cd": Quantity(Dimension.from_mapping({"J": 1})),
    "N": Quantity(Dimension.from_mapping({"M": 1, "L": 1, "T": -2})),
    "Pa": Quantity(Dimension.from_mapping({"M": 1, "L": -1, "T": -2})),
    "J": Quantity(Dimension.from_mapping({"M": 1, "L": 2, "T": -2})),
    "W": Quantity(Dimension.from_mapping({"M": 1, "L": 2, "T": -3})),
    # Energy as billed: one watt-hour is 3600 joules.
    "Wh": Quantity(Dimension.from_mapping({"M": 1, "L": 2, "T": -2}), 3600.0),
    # Currency. "E" is Euro here, which is why exa is not offered as a prefix.
    "E": Quantity(Dimension.from_mapping({"Currency": 1})),
    "EUR": Quantity(Dimension.from_mapping({"Currency": 1})),
    "€": Quantity(Dimension.from_mapping({"Currency": 1})),
    "USD": Quantity(Dimension.from_mapping({"Currency": 1})),
    "$": Quantity(Dimension.from_mapping({"Currency": 1})),
}

#: Decimal prefixes. ``E`` (exa), ``P``, ``Z``, ``Y`` and ``h`` (hecto) are
#: deliberately absent: ``E`` and ``h`` are taken by Euro and hour, and the rest
#: never appear in a parameter table but would add ways to mistype one.
SI_PREFIXES: dict[str, float] = {
    "n": 1e-9,
    "u": 1e-6,
    "µ": 1e-6,
    "m": 1e-3,
    "c": 1e-2,
    "d": 1e-1,
    "k": 1e3,
    "M": 1e6,
    "G": 1e9,
    "T": 1e12,
}


def _resolve_symbol(symbol: str) -> Quantity:
    """One symbol, with an optional decimal prefix, as a quantity."""
    if symbol in UNIT_SYMBOLS:
        return UNIT_SYMBOLS[symbol]
    for prefix, factor in SI_PREFIXES.items():
        if symbol.startswith(prefix) and len(symbol) > len(prefix):
            base = symbol[len(prefix):]
            if base in UNIT_SYMBOLS:
                unit = UNIT_SYMBOLS[base]
                return Quantity(unit.dimension, unit.scale * factor)
    raise ValueError(
        f"Unknown unit symbol {symbol!r}. Known symbols: {sorted(UNIT_SYMBOLS)}; "
        f"decimal prefixes: {sorted(SI_PREFIXES)}"
    )


def parse_quantity(text: str) -> Quantity:
    """Parse a flat unit string into a dimension and a scale.

    Accepted forms are symbols joined by ``*`` and ``/``, each optionally raised
    to an integer or fractional power with ``^``, and each optionally carrying a
    decimal prefix.  A bare number is a scale factor, so ``1000*W`` and ``kW``
    parse to the same quantity.  Examples: ``W/m/K``, ``J/m^3/K``, ``kW``,
    ``E/kWh``, ``cE/kWh``, ``1``.

    Only the flat form is accepted -- no brackets -- because a silently
    mis-parsed unit would corrupt every Pi group derived from it, and a
    bracketing mistake is the easy way to produce one.
    """
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Empty unit string.")

    tokens: list[tuple[str, str]] = []
    current = ""
    operator = "*"
    for character in cleaned:
        if character in "*/":
            tokens.append((operator, current))
            operator = character
            current = ""
        else:
            current += character
    tokens.append((operator, current))

    result = Quantity(DIMENSIONLESS, 1.0)
    for token_operator, token in tokens:
        token = token.strip()
        if not token:
            raise ValueError(f"Malformed unit string {text!r}.")
        if "^" in token:
            symbol, _, power_text = token.partition("^")
            try:
                power = Fraction(power_text)
            except (ValueError, ZeroDivisionError) as error:
                raise ValueError(f"Bad exponent in {token!r}.") from error
        else:
            symbol, power = token, Fraction(1)
        symbol = symbol.strip()

        try:
            # A bare number is a pure scale factor, so "1000*W" works.
            contribution = Quantity(DIMENSIONLESS, float(symbol))
        except ValueError:
            contribution = _resolve_symbol(symbol)

        contribution = contribution ** power
        result = result * contribution if token_operator == "*" else result / contribution
    return result


def parse_unit(text: str) -> Dimension:
    """Dimension of a unit string, ignoring its scale."""
    return parse_quantity(text).dimension


def unit_scale(text: str) -> float:
    """Size of a unit string relative to the coherent SI unit of its dimension."""
    return parse_quantity(text).scale


def convert_factor(source: str, target: str) -> float:
    """Multiplier converting a value from *source* units to *target* units.

    Raises
    ------
    ValueError
        If the two units do not share a dimension, which is a conversion that
        cannot be meant.
    """
    a = parse_quantity(source)
    b = parse_quantity(target)
    if a.dimension != b.dimension:
        raise ValueError(
            f"Cannot convert {source!r} to {target!r}: dimensions {a.dimension} "
            f"and {b.dimension} differ."
        )
    return a.scale / b.scale


def dimension_matrix(dimensions: Iterable[Dimension]) -> list[list[Fraction]]:
    """Rows are base dimensions, columns are quantities."""
    columns = [dimension.exponents for dimension in dimensions]
    return [
        [column[row] for column in columns] for row in range(len(BASE_DIMENSIONS))
    ]
