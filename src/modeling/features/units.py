"""Minimal SI dimensional algebra.

Only what dimensional analysis of a simulator study needs: a vector of base-unit
exponents, multiplication, division and integer powers, plus a parser for flat
unit strings such as ``"W/m/K"`` or ``"J/m^3/K"``.

The parser deliberately accepts only the flat form -- symbols joined by ``*``
and ``/``, each optionally raised to an integer power with ``^`` -- because that
covers every quantity that appears in a parameter table and leaves no room for
bracketing mistakes to pass silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Mapping

#: Base dimensions, in the order used by :attr:`Dimension.exponents`.
BASE_DIMENSIONS: tuple[str, ...] = ("M", "L", "T", "Theta", "I", "N", "J")

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

#: Unit symbols understood by :func:`parse_unit`.
UNIT_SYMBOLS: dict[str, Dimension] = {
    "1": DIMENSIONLESS,
    "-": DIMENSIONLESS,
    "kg": Dimension.from_mapping({"M": 1}),
    "m": Dimension.from_mapping({"L": 1}),
    "s": Dimension.from_mapping({"T": 1}),
    "K": Dimension.from_mapping({"Theta": 1}),
    "A": Dimension.from_mapping({"I": 1}),
    "mol": Dimension.from_mapping({"N": 1}),
    "cd": Dimension.from_mapping({"J": 1}),
    "N": Dimension.from_mapping({"M": 1, "L": 1, "T": -2}),
    "Pa": Dimension.from_mapping({"M": 1, "L": -1, "T": -2}),
    "J": Dimension.from_mapping({"M": 1, "L": 2, "T": -2}),
    "W": Dimension.from_mapping({"M": 1, "L": 2, "T": -3}),
}


def parse_unit(text: str) -> Dimension:
    """Parse a flat unit string such as ``"W/m/K"`` or ``"J/m^3/K"``.

    Raises
    ------
    ValueError
        If a symbol is unknown or the expression is malformed.  Failing loudly
        matters here: a silently mis-parsed unit would corrupt every Pi group
        derived from it.
    """
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Empty unit string.")

    tokens: list[tuple[str, str]] = []  # (operator, token)
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

    result = DIMENSIONLESS
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
        if symbol not in UNIT_SYMBOLS:
            raise ValueError(
                f"Unknown unit symbol {symbol!r} in {text!r}. "
                f"Known symbols: {sorted(UNIT_SYMBOLS)}"
            )
        contribution = UNIT_SYMBOLS[symbol] ** power
        result = result * contribution if token_operator == "*" else result / contribution
    return result


def dimension_matrix(dimensions: Iterable[Dimension]) -> list[list[Fraction]]:
    """Rows are base dimensions, columns are quantities."""
    columns = [dimension.exponents for dimension in dimensions]
    return [
        [column[row] for column in columns] for row in range(len(BASE_DIMENSIONS))
    ]
