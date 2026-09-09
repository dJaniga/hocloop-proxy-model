"""Experiment driver for the structured proxy model.

Runs the full study and writes every artefact a paper needs into the output
directory: the discovered target structure, the dimensional analysis, the
benchmark and ablation tables, the learning curve, conformal coverage, the
analytic design rules and the final closed-form model.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from modeling.benchmarks import ablation_study, learner_comparison, to_dataframe
from modeling.conformal import (
    LogScaleConformalCalibrator,
    coverage_report,
    cross_validated_log_scores,
    propagate_interval,
)
from modeling.design_rules import (
    length_exponents_by_variable,
    optimal_length_rule,
    sensitivity_table,
)
from modeling.learners import LEARNER_FACTORIES, build_learner_factory
from modeling.proxy import ProxyConfig, StructuredProxyModel
from modeling.transforms import columns_from_matrix
from modeling.validation import learning_curve

logger = logging.getLogger(__name__)


#: Units for the HOCLOOP closed-loop geothermal parameter study.  ``u_inf`` is a
#: base-10 logarithm of a groundwater velocity and is therefore dimensionless.
DEFAULT_FEATURE_UNITS: dict[str, str] = {
    "k_rock": "W/m/K",
    "rho_cp_rock": "J/m^3/K",
    "gradT": "K/m",
    "u_inf": "1",
    "mass_flow": "kg/s",
    "depth": "m",
    "l_horiz": "m",
}

#: ``w_out`` is a thermal power; the levelised costs are reported per unit
#: energy and carry no dimension in this parameter set.
DEFAULT_TARGET_UNITS: dict[str, str] = {
    "w_out": "W",
    "LCOH": "1",
    "LCOH_i": "1",
}


@dataclass
class ExperimentSettings:
    """What to run and how hard."""

    learners: tuple[str, ...] = (
        "power_law_ols",
        "symbolic_deap",
        "symbolic_residual",
        "symbolic_pysr",
        "polynomial_ridge",
        "gradient_boosting",
        "random_forest",
        "neural_network",
    )
    primary_learner: str = "symbolic_residual"
    cv_splits: int = 5
    cv_repeats: int = 1
    shell_fraction: float = 0.25
    conformal_alpha: float = 0.1
    conformal_splits: int = 5
    learning_curve_sizes: tuple[int, ...] = (100, 200, 400, 800, 1600, 3200)
    learning_curve_repeats: int = 2
    run_benchmarks: bool = True
    run_ablation: bool = True
    run_learning_curve: bool = True
    run_conformal: bool = True
    run_design_rules: bool = True
    seed: int | None = 0
    symbolic_overrides: dict[str, Any] = field(default_factory=dict)


def load_data(file: Path) -> pd.DataFrame:
    logger.info("Loading data from %s", file)
    return pd.read_csv(file)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    logger.info("Wrote %s", path)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    logger.info("Wrote %s", path)


def build_config(
    feature_names: Sequence[str],
    target_names: Sequence[str],
    units_file: Path | None = None,
) -> ProxyConfig:
    """Assemble a :class:`ProxyConfig`, reading units from a file when given.

    The units file is JSON with ``features`` and ``targets`` objects mapping a
    column name to a flat unit string.  Without units the dimensional stage is
    switched off rather than guessed at, because a wrong dimension would produce
    a confidently meaningless Pi group.
    """
    feature_units = dict(DEFAULT_FEATURE_UNITS)
    target_units = dict(DEFAULT_TARGET_UNITS)

    if units_file is not None:
        payload = json.loads(Path(units_file).read_text(encoding="utf-8"))
        feature_units = dict(payload.get("features", {}))
        target_units = dict(payload.get("targets", {}))
        logger.info("Loaded units from %s", units_file)

    missing = [name for name in feature_names if name not in feature_units]
    if missing:
        logger.warning(
            "No units for %s; dimensional reduction disabled. Supply --units-file "
            "to enable it.",
            missing,
        )
        return ProxyConfig(
            feature_units={},
            target_units={},
            use_dimensional_reduction=False,
            feature_space="raw",
        )

    return ProxyConfig(
        feature_units={name: feature_units[name] for name in feature_names},
        target_units={name: target_units[name] for name in target_names if name in target_units},
    )


def _symbolic_factory(settings: ExperimentSettings) -> Callable[[], Any]:
    """Factory for the learner used as the reference model everywhere else.

    The ablation, the learning curve, the conformal calibration and the reported
    closed form all use this one learner, so they describe the same model.
    """
    return build_learner_factory(settings.primary_learner, **settings.symbolic_overrides)


def _learner_factories(settings: ExperimentSettings) -> dict[str, Callable[[], Any]]:
    factories: dict[str, Callable[[], Any]] = {}
    for name in settings.learners:
        if name not in LEARNER_FACTORIES:
            logger.warning("Unknown learner %r; skipping.", name)
            continue
        symbolic = {"symbolic_deap", "symbolic_residual"}
        factories[name] = (
            build_learner_factory(name, **settings.symbolic_overrides)
            if name in symbolic
            else build_learner_factory(name)
        )
    return factories


def _run_conformal(
    settings: ExperimentSettings,
    config: ProxyConfig,
    learner_factory: Callable[[], Any],
    features: np.ndarray,
    targets: np.ndarray,
    feature_names: Sequence[str],
    target_names: Sequence[str],
) -> dict[str, Any]:
    """Calibrate a log-scale conformal radius and propagate it to every target."""
    def model_factory() -> StructuredProxyModel:
        return StructuredProxyModel(learner_factory=learner_factory, config=config)

    fitted = model_factory()
    fitted.fit(features, targets, tuple(feature_names), tuple(target_names))
    independent = fitted.structure_.independent_targets  # type: ignore[union-attr]

    predictions = np.asarray(fitted.predict(features), dtype=np.float64)
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
    point = {name: predictions[:, index] for index, name in enumerate(target_names)}
    actual = {name: targets[:, index] for index, name in enumerate(target_names)}

    intervals = {}
    radii: dict[str, float] = {}
    for target in independent:
        scores = cross_validated_log_scores(
            model_factory, features, targets, feature_names, target_names, target,
            n_splits=settings.conformal_splits, seed=settings.seed,
        )
        calibrator = LogScaleConformalCalibrator(
            alpha=settings.conformal_alpha, n_splits=settings.conformal_splits,
            seed=settings.seed,
        ).calibrate_from_scores(scores)
        radii[target] = calibrator.radius_
        intervals[target] = calibrator.interval(point[target])

    columns = fitted.prepare_columns(features)
    all_intervals = propagate_interval(fitted.structure_, intervals, point, columns)  # type: ignore[arg-type]
    return {
        "alpha": settings.conformal_alpha,
        "log_radius": radii,
        "multiplicative_factor": {k: float(np.exp(v)) for k, v in radii.items()},
        "coverage": coverage_report(all_intervals, actual, point),
        "note": (
            "Intervals for derived targets are obtained by evaluating the exact "
            "identity at the endpoints of the independent target's interval, so "
            "they inherit its coverage without an assumed error budget."
        ),
    }


def _run_design_rules(
    model: StructuredProxyModel,
    features: np.ndarray,
) -> dict[str, Any]:
    """Differentiate the fitted closed form to get optimal-design formulas.

    A separate rule is produced for each section of the well that could be
    extended, because the output responds differently to a deeper well than to a
    longer horizontal one and a single "length" rule conflates the two.
    """
    if model.structure_ is None:
        return {}
    columns = model.prepare_columns(features)
    rules: dict[str, Any] = {}

    for name, target_model in model.target_models_.items():
        if target_model.pi_basis is None:
            continue
        scale = target_model.pi_basis.scale_exponents
        rules.setdefault("sensitivity", {})[name] = sensitivity_table(scale)

        aggregates = [
            tuple(aggregate)
            for identity in model.structure_.identities.values()
            for aggregate in identity.parameters.get("aggregates", [])
            if len(aggregate) > 1
        ]
        if not aggregates:
            continue
        members = list(dict.fromkeys(aggregates[0]))

        exponents = length_exponents_by_variable(scale, members)
        rules.setdefault("length_exponent_by_variable", {})[name] = exponents

        total = np.zeros(features.shape[0], dtype=np.float64)
        for member in members:
            total = total + np.asarray(columns[member], dtype=np.float64)
        sampled = (float(total.min()), float(total.max()))
        rules.setdefault("sampled_total_length", {})[name] = list(sampled)

        for derived in model.structure_.derived_targets:
            for variable, exponent in exponents.items():
                if exponent <= 0:
                    # The rule solves d/dL[log C(L) - p log L] = 0 for a cost
                    # divided by an output that grows with length.  A
                    # non-positive exponent means the learnt target is not that
                    # output -- typically because a cost was learnt and the
                    # output derived -- and the stationary point would be
                    # meaningless.
                    logger.info(
                        "Skipping length rule for %r via %r: output exponent %.3f "
                        "is not positive, so %r is not an output that grows with "
                        "drilled length.",
                        derived, variable, exponent, name,
                    )
                    continue
                rule = optimal_length_rule(
                    model.structure_,
                    derived,
                    exponent,
                    design_variable=variable,
                    sampled_range=sampled,
                )
                if rule is not None:
                    rules.setdefault("optimal_length", {}).setdefault(derived, {})[
                        variable
                    ] = rule.to_dict()
    return rules


def pipeline(
    feature_file_path: Path,
    target_file_path: Path,
    output_path: Path,
    units_file: Path | None = None,
    settings: ExperimentSettings | None = None,
) -> StructuredProxyModel:
    """Run the whole study and write every artefact to *output_path*."""
    settings = settings or ExperimentSettings()
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    feature_frame = load_data(feature_file_path)
    target_frame = load_data(target_file_path)
    if len(feature_frame) != len(target_frame):
        raise ValueError(
            f"Feature file has {len(feature_frame)} rows but target file has "
            f"{len(target_frame)}; they must correspond row by row."
        )

    feature_names = tuple(feature_frame.columns)
    target_names = tuple(target_frame.columns)
    features = feature_frame.to_numpy(dtype=np.float64)
    targets = target_frame.to_numpy(dtype=np.float64)

    config = build_config(feature_names, target_names, units_file)
    symbolic_factory = _symbolic_factory(settings)

    # ---------------------------------------------------------------- #
    # Final model on all data: structure, dimensional analysis, formula  #
    # ---------------------------------------------------------------- #
    logger.info("Fitting the reference model on the full design.")
    model = StructuredProxyModel(learner_factory=symbolic_factory, config=config)
    model.fit(features, targets, feature_names, target_names)

    details = model.get_fit_details()
    _write_json(output_path / "target_structure.json", details["structure"])
    _write_json(
        output_path / "dimensional_analysis.json",
        [entry["pi"] for entry in details["target_models"]],
    )
    _write_json(
        output_path / "final_model.json",
        {"details": details, "closed_form": model.closed_form()},
    )

    predictions = np.asarray(model.predict(features), dtype=np.float64)
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
    prediction_frame = pd.DataFrame(
        {name: features[:, index] for index, name in enumerate(feature_names)}
    )
    for index, name in enumerate(target_names):
        prediction_frame[f"actual_{name}"] = targets[:, index]
        prediction_frame[f"predicted_{name}"] = predictions[:, index]
    _write_csv(output_path / "predictions_full_fit.csv", prediction_frame)

    # ---------------------------------------------------------------- #
    # Benchmarks, ablation, learning curve, conformal, design rules      #
    # ---------------------------------------------------------------- #
    common = {
        "n_splits": settings.cv_splits,
        "n_repeats": settings.cv_repeats,
        "shell_fraction": settings.shell_fraction,
        "seed": settings.seed,
    }

    if settings.run_benchmarks:
        entries = learner_comparison(
            _learner_factories(settings), config, features, targets,
            feature_names, target_names, **common,
        )
        _write_json(output_path / "benchmark_learners.json", [e.to_dict() for e in entries])
        _write_csv(output_path / "benchmark_learners.csv", to_dataframe(entries))

    if settings.run_ablation:
        entries = ablation_study(
            symbolic_factory, config, features, targets,
            feature_names, target_names, **common,
        )
        _write_json(output_path / "ablation.json", [e.to_dict() for e in entries])
        _write_csv(output_path / "ablation.csv", to_dataframe(entries))

    if settings.run_learning_curve:
        def model_factory() -> StructuredProxyModel:
            return StructuredProxyModel(learner_factory=symbolic_factory, config=config)

        curve = learning_curve(
            model_factory, features, targets, feature_names, target_names,
            train_sizes=settings.learning_curve_sizes,
            n_repeats=settings.learning_curve_repeats,
            seed=settings.seed,
        )
        rows = [
            {
                "repeat": result.fold,
                "n_train": result.n_train,
                "target": target,
                "r2_score": values["r2_score"],
                "r2_log_score": values["r2_log_score"],
                "mean_absolute_percentage_error": values["mean_absolute_percentage_error"],
            }
            for result in curve
            for target, values in result.metrics.items()
        ]
        _write_csv(output_path / "learning_curve.csv", pd.DataFrame(rows))

    if settings.run_conformal:
        report = _run_conformal(
            settings, config, symbolic_factory, features, targets,
            feature_names, target_names,
        )
        _write_json(output_path / "conformal_intervals.json", report)

    if settings.run_design_rules:
        _write_json(output_path / "design_rules.json", _run_design_rules(model, features))

    logger.info("Study complete. Artefacts written to %s", output_path)
    for name, formula in model.closed_form().items():
        logger.info("  %s = %s", name, formula)
    return model
