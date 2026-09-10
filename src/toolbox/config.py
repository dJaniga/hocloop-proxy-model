"""Experiment configuration: which columns to use, and in what units.

One file declares both the column selection and the units, because the two are
the same decision seen from different angles: dropping a feature and failing to
give it a unit both remove it from the dimensional stage, and having them in
separate places is how they drift apart.

The file may be YAML or JSON; the format is chosen by extension, with YAML the
default for an unrecognised one. The full form is::

    features:
      include: [k_rock, gradT, mass_flow, depth, l_horiz]
      exclude: [u_inf]
      units:
        k_rock: W/m/K
        depth: m
    targets:
      include: [w_out, LCOH]
      units:
        w_out: kW
        LCOH: cE/kWh

``include`` selects columns and fixes their order; omit it to take every column
in the data file. ``exclude`` removes columns from whatever ``include`` left.
Naming a column that the data does not contain is an error rather than a
silently ignored line, since a typo there would otherwise quietly change the
experiment.

The older shape, in which ``features`` and ``targets`` map a name straight to a
unit string, is still accepted and understood as units with no selection.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)


@dataclass
class ColumnSelection:
    """Which columns to use from one file, and their declared units."""

    include: tuple[str, ...] | None = None
    exclude: tuple[str, ...] = ()
    units: dict[str, str] = field(default_factory=dict)

    def resolve(self, available: Sequence[str], role: str) -> tuple[str, ...]:
        """Column names to use, in order.

        Raises
        ------
        ValueError
            If a named column is absent, or if the selection is empty.
        """
        available = tuple(available)
        if self.include is None:
            chosen = list(available)
        else:
            missing = [name for name in self.include if name not in available]
            if missing:
                raise ValueError(
                    f"{role} include lists {missing!r}, which the data file does not "
                    f"contain. Available: {list(available)}"
                )
            chosen = list(self.include)

        unknown_excludes = [name for name in self.exclude if name not in available]
        if unknown_excludes:
            raise ValueError(
                f"{role} exclude lists {unknown_excludes!r}, which the data file does "
                f"not contain. Available: {list(available)}"
            )
        chosen = [name for name in chosen if name not in set(self.exclude)]

        if not chosen:
            raise ValueError(f"The {role} selection leaves no columns.")
        return tuple(chosen)

    def units_for(self, chosen: Sequence[str]) -> dict[str, str]:
        return {name: self.units[name] for name in chosen if name in self.units}


@dataclass
class ExperimentConfigFile:
    """Parsed contents of a configuration file."""

    features: ColumnSelection = field(default_factory=ColumnSelection)
    targets: ColumnSelection = field(default_factory=ColumnSelection)
    source: Path | None = None

    def describe(self) -> dict[str, Any]:
        return {
            "source": str(self.source) if self.source else None,
            "features": {
                "include": list(self.features.include) if self.features.include else None,
                "exclude": list(self.features.exclude),
                "units": dict(self.features.units),
            },
            "targets": {
                "include": list(self.targets.include) if self.targets.include else None,
                "exclude": list(self.targets.exclude),
                "units": dict(self.targets.units),
            },
        }


def _parse_section(payload: Any, role: str) -> ColumnSelection:
    if payload is None:
        return ColumnSelection()
    if not isinstance(payload, dict):
        raise ValueError(f"The {role!r} section must be a mapping, got {type(payload).__name__}.")

    known_keys = {"include", "exclude", "units", "use", "drop"}
    if not (set(payload) & known_keys):
        # Legacy shape: the section is itself a name-to-unit mapping.
        return ColumnSelection(units={str(k): str(v) for k, v in payload.items()})

    include = payload.get("include", payload.get("use"))
    exclude = payload.get("exclude", payload.get("drop", []))
    units = payload.get("units", {}) or {}
    if include is not None and not isinstance(include, (list, tuple)):
        raise ValueError(f"{role}.include must be a list.")
    if not isinstance(exclude, (list, tuple)):
        raise ValueError(f"{role}.exclude must be a list.")
    if not isinstance(units, dict):
        raise ValueError(f"{role}.units must be a mapping.")

    return ColumnSelection(
        include=tuple(str(name) for name in include) if include is not None else None,
        exclude=tuple(str(name) for name in exclude),
        units={str(k): str(v) for k, v in units.items()},
    )


def load_config_file(path: Path | None) -> ExperimentConfigFile:
    """Read a YAML or JSON configuration file."""
    if path is None:
        return ExperimentConfigFile()

    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
    else:
        import yaml

        payload = yaml.safe_load(text)

    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a mapping at the top level.")

    # Keys beginning with an underscore are comments in the JSON files that
    # predate YAML support.
    payload = {k: v for k, v in payload.items() if not str(k).startswith("_")}

    unexpected = set(payload) - {"features", "targets"}
    if unexpected:
        logger.warning("Ignoring unrecognised configuration sections: %s", sorted(unexpected))

    config = ExperimentConfigFile(
        features=_parse_section(payload.get("features"), "features"),
        targets=_parse_section(payload.get("targets"), "targets"),
        source=path,
    )
    logger.info("Loaded configuration from %s", path)
    return config


def validate_units(units: dict[str, str], role: str) -> None:
    """Fail loudly on an unparsable unit, naming the column."""
    from modeling.features.units import parse_quantity

    for name, unit in units.items():
        try:
            parse_quantity(unit)
        except ValueError as error:
            raise ValueError(f"{role} column {name!r} has an invalid unit {unit!r}: {error}") from error
