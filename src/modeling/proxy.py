"""The structured proxy model.

The model composes four stages, each of which is fitted from training data only,
so the whole object can be dropped into a cross-validation loop without leaking
information between folds:

1. **Target-space structure discovery.**  Exact algebraic relations between the
   targets are found and only the irreducible targets are learnt.  The rest are
   reconstructed in closed form, which makes the proxy satisfy those relations
   by construction rather than approximately.
2. **Dimensional reduction.**  A Buckingham-Pi basis is chosen from the data and
   its response group is refined to the best dimensionally consistent power law.
3. **Regression of the residual.**  A pluggable learner fits the dimensionless
   departure from that power law on a log scale.  Swapping the learner while
   holding stages 1, 2 and 4 fixed is what makes the benchmark table an ablation
   rather than a comparison of unrelated pipelines.
4. **Reconstruction.**  Predictions of the independent targets are expanded to
   the full target set through the discovered identities.

Every stage can be switched off, which is how the ablation study isolates the
contribution of each one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

from modeling.base import Regressor
from modeling.features.dimensional import (
    PiBasis,
    augment_with_derived_variables,
    prune_redundant_groups,
    refine_response_group,
    select_pi_basis,
)
from modeling.structure.discovery import DiscoveryConfig, discover_target_structure
from modeling.structure.identities import TargetStructure
from modeling.transforms import (
    DimensionlessLogResponse,
    FeatureSpace,
    IdentityResponse,
    LogResponse,
    ResponseTransform,
    columns_from_matrix,
)

logger = logging.getLogger(__name__)


def clip_to_observed_range(
    values: np.ndarray,
    observed: tuple[float, float] | None,
    margin: float,
) -> np.ndarray:
    """Hold predictions near the range a target was observed to take.

    A positive target is modelled on a log scale, so the margin widens the range
    multiplicatively; anything else gets an additive margin on the observed
    width.  Returning the edge of the fitted range is more useful than returning
    a value the simulator could never produce.
    """
    if observed is None or not np.isfinite(margin):
        return values
    low, high = observed
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return values
    if low > 0:
        return np.clip(values, low / margin, high * margin)
    width = high - low
    return np.clip(values, low - margin * width, high + margin * width)


class BaseLearner(Protocol):
    """Minimal scikit-learn-style interface required of a stage-3 learner."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> Any: ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...


LearnerFactory = Callable[[], BaseLearner]


@dataclass
class ProxyConfig:
    """Switches controlling which stages of the pipeline are active."""

    #: Physical units of each feature, as flat unit strings such as ``"W/m/K"``.
    feature_units: dict[str, str] = field(default_factory=dict)
    #: Physical unit of every target, keyed by target name.
    target_units: dict[str, str] = field(default_factory=dict)
    #: Stage 1: search the target space for exact identities.
    use_structure_discovery: bool = True
    #: Stage 2: reduce to dimensionless groups before regressing.
    use_dimensional_reduction: bool = True
    #: Stage 2b: absorb the best power law into the response group.
    refine_power_law: bool = True
    #: Stage 3: regress on a log scale (with Duan smearing on the inverse).
    log_response: bool = True
    #: Feature space handed to the learner: ``"log_pi_plus_raw"``, ``"log_pi"``,
    #: ``"pi"`` or ``"raw"``.  The default keeps the raw features alongside the
    #: dimensionless groups, because strict similarity holds only when every
    #: governing dimensional quantity appears in the parameter table.
    feature_space: str = "log_pi_plus_raw"
    #: Fixed physical quantities to admit into the Pi construction.
    physical_constants: dict[str, tuple[float, str]] = field(default_factory=dict)
    #: How far outside the training range of the target a prediction may go, as
    #: a multiplicative factor for a positive target.  The response is modelled
    #: on a log scale and exponentiated on the way out, so an unchecked
    #: extrapolation can be wrong by orders of magnitude and, through the
    #: reconstruction identities, corrupt every target at once.  Set to
    #: ``float("inf")`` to disable.
    prediction_range_margin: float = 10.0
    #: Settings for the structure search.
    discovery: DiscoveryConfig = field(default_factory=DiscoveryConfig)

    def units_for(self, target: str) -> str:
        try:
            return self.target_units[target]
        except KeyError as error:
            raise KeyError(
                f"No unit declared for target {target!r}; dimensional reduction "
                "needs one. Provide it in ProxyConfig.target_units or disable "
                "use_dimensional_reduction."
            ) from error


@dataclass
class _TargetModel:
    """Everything needed to predict one independent target."""

    name: str
    learner: BaseLearner
    response: ResponseTransform
    feature_space: FeatureSpace
    pi_basis: PiBasis | None
    pi_report: dict[str, Any]
    latent_range: tuple[float, float] | None = None
    target_range: tuple[float, float] | None = None
    prediction_margin: float = 10.0

    def _clip_prediction(self, values: np.ndarray) -> np.ndarray:
        """Hold the physical prediction near the range the target was trained on.

        The model predicts a log quantity that the inverse transform
        exponentiates, so a moderate error in that latent becomes a factor of
        many orders of magnitude in the output.  This is not hypothetical: on a
        small training subset the dimensional stage can settle on a poor Pi
        basis whose response group spans several log units, and the fitted model
        then emits values from ``1e-4`` to ``3e6`` for a target whose observed
        range is ``5`` to ``2.3e4``.

        Bounding the latent instead is not enough, because that bound is derived
        from the very spread that has gone wrong.  The bound that means
        something is the observed range of the target itself.
        """
        return clip_to_observed_range(values, self.target_range, self.prediction_margin)

    def predict(self, columns: Mapping[str, np.ndarray]) -> np.ndarray:
        design = self.feature_space.transform(columns)
        latent = np.asarray(self.learner.predict(design), dtype=np.float64).ravel()
        return self._clip_prediction(self.response.inverse(columns, latent))

    def describe(self) -> dict[str, Any]:
        description: dict[str, Any] = {
            "target": self.name,
            "response_transform": self.response.describe(),
            "feature_space": self.feature_space.describe(),
            "pi": self.pi_report,
            "latent_range": list(self.latent_range) if self.latent_range else None,
            "target_range": list(self.target_range) if self.target_range else None,
        }
        expression = getattr(self.learner, "expression_", None)
        if expression is not None:
            description["expression"] = expression
        details = getattr(self.learner, "fit_details_", None)
        if details is not None:
            description["learner_details"] = details
        return description


@dataclass
class StructuredProxyModel(Regressor):
    """Composite proxy: structure discovery, Pi reduction, learner, reconstruction."""

    learner_factory: LearnerFactory
    config: ProxyConfig = field(default_factory=ProxyConfig)
    name: str = "structured_proxy"

    # --- fitted state -----------------------------------------------------
    structure_: TargetStructure | None = None
    target_models_: dict[str, _TargetModel] = field(default_factory=dict)
    features_name_: tuple[str, ...] = ()
    targets_name_: tuple[str, ...] = ()
    derived_variables_: dict[str, tuple[tuple[str, ...], str]] = field(default_factory=dict)
    target_ranges_: dict[str, tuple[float, float]] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.name

    # ------------------------------------------------------------------ #
    # Fitting                                                             #
    # ------------------------------------------------------------------ #

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        features_name: tuple[str, ...] | None = None,
        targets_name: tuple[str, ...] | None = None,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> StructuredProxyModel:
        features = np.asarray(features, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.float64)
        if targets.ndim == 1:
            targets = targets.reshape(-1, 1)

        self.features_name_ = tuple(
            features_name or tuple(f"ARG{i}" for i in range(features.shape[1]))
        )
        self.targets_name_ = tuple(
            targets_name or tuple(f"target_{k}" for k in range(targets.shape[1]))
        )
        columns = columns_from_matrix(features, self.features_name_)
        target_columns = {
            name: targets[:, index] for index, name in enumerate(self.targets_name_)
        }

        self.structure_ = self._discover_structure(features, targets)
        independent = self.structure_.independent_targets
        self.derived_variables_ = self._aggregates_from_structure(self.structure_)

        # Observed range of every target, including the derived ones, so that a
        # reconstruction cannot carry a runaway that the learnt target's own
        # bound never sees: the identities divide, and division turns a small
        # predicted value into an enormous derived one.
        self.target_ranges_ = {
            name: (float(np.min(values)), float(np.max(values)))
            for name, values in target_columns.items()
        }

        self.target_models_ = {}
        for target in independent:
            self.target_models_[target] = self._fit_one_target(
                target, columns, target_columns[target]
            )
        return self

    def _discover_structure(
        self, features: np.ndarray, targets: np.ndarray
    ) -> TargetStructure:
        if not self.config.use_structure_discovery or targets.shape[1] < 2:
            return TargetStructure(
                target_names=self.targets_name_,
                independent_targets=self.targets_name_,
                tolerance=self.config.discovery.tolerance,
            )
        unit_groups = {
            name: self.config.feature_units.get(name, "?") for name in self.features_name_
        }
        return discover_target_structure(
            features,
            targets,
            self.features_name_,
            self.targets_name_,
            config=self.config.discovery,
            unit_groups=unit_groups if self.config.feature_units else None,
        )

    @staticmethod
    def _aggregates_from_structure(
        structure: TargetStructure,
    ) -> dict[str, tuple[tuple[str, ...], str]]:
        """Reuse feature aggregates found in target space as modelling variables.

        A group of features the identity search had to sum before a target became
        a polynomial -- a total drilled length, say -- is a physically meaningful
        variable in its own right, so it is offered to the dimensional stage.
        """
        aggregates: dict[str, tuple[tuple[str, ...], str]] = {}
        for identity in structure.identities.values():
            for members in identity.parameters.get("aggregates", []):
                if len(members) < 2:
                    continue
                name = "_plus_".join(members)
                aggregates[name] = (tuple(members), "")
        return aggregates

    def _fit_one_target(
        self,
        target: str,
        columns: Mapping[str, np.ndarray],
        y: np.ndarray,
    ) -> _TargetModel:
        working_columns = dict(columns)
        pi_basis: PiBasis | None = None
        pi_report: dict[str, Any] = {"enabled": False}

        if self.config.use_dimensional_reduction:
            pi_basis, pi_report, working_columns = self._build_pi_basis(
                target, working_columns, y
            )

        if pi_basis is not None:
            response: ResponseTransform = (
                DimensionlessLogResponse(pi_basis=pi_basis)
                if self.config.log_response
                else IdentityResponse()
            )
            space = FeatureSpace(
                kind=self.config.feature_space,
                pi_basis=pi_basis,
                raw_names=self.features_name_,
            )
        else:
            response = LogResponse() if self.config.log_response else IdentityResponse()
            space = FeatureSpace(kind="raw", raw_names=self.features_name_)

        response.fit(working_columns, y)
        space.fit_columns(working_columns)
        design = space.transform(working_columns)
        latent = response.forward(working_columns, y)

        learner = self.learner_factory()
        # Symbolic learners print their expression in terms of the column names,
        # so a formula reads in Pi groups rather than in x0, x1, x2.
        if hasattr(learner, "feature_names"):
            learner.feature_names = tuple(space.names)
        learner.fit(design, latent)
        # Calibrate the back-transform bias on in-sample predictions, which is
        # where Duan's estimator is defined.
        response.set_residual_correction(
            working_columns, y, np.asarray(learner.predict(design), dtype=np.float64).ravel()
        )

        logger.info(
            "Fitted learner for %r on %d features in space %r",
            target,
            design.shape[1],
            space.kind,
        )
        return _TargetModel(
            name=target,
            learner=learner,
            response=response,
            feature_space=space,
            pi_basis=pi_basis,
            pi_report=pi_report,
            latent_range=(float(np.min(latent)), float(np.max(latent))),
            target_range=(float(np.min(y)), float(np.max(y))),
            prediction_margin=self.config.prediction_range_margin,
        )

    def _build_pi_basis(
        self,
        target: str,
        columns: dict[str, np.ndarray],
        y: np.ndarray,
    ) -> tuple[PiBasis | None, dict[str, Any], dict[str, np.ndarray]]:
        units = dict(self.config.feature_units)
        missing = [name for name in self.features_name_ if name not in units]
        if missing:
            logger.warning(
                "No units for %s; skipping dimensional reduction for %r.", missing, target
            )
            return None, {"enabled": False, "reason": f"missing units for {missing}"}, columns

        columns = {name: columns[name] for name in self.features_name_}
        units = {name: units[name] for name in self.features_name_}

        derived = {
            name: (members, units[members[0]])
            for name, (members, _) in self.derived_variables_.items()
            if all(member in units for member in members)
        }
        if derived:
            columns, units = augment_with_derived_variables(columns, units, derived)
        if self.config.physical_constants:
            from modeling.features.dimensional import add_physical_constants

            columns, units = add_physical_constants(
                columns, units, self.config.physical_constants
            )

        try:
            basis, _ = select_pi_basis(
                target,
                units,
                self.config.units_for(target),
                columns,
                y,
                feature_order=tuple(units),
            )
        except (ValueError, KeyError) as error:
            logger.warning("Pi reduction unavailable for %r: %s", target, error)
            return None, {"enabled": False, "reason": str(error)}, columns

        basis = prune_redundant_groups(basis, columns)
        report: dict[str, Any] = {"enabled": True}
        if self.config.refine_power_law:
            basis, refinement = refine_response_group(basis, columns, y)
            report["refinement"] = refinement
        report["basis"] = basis.to_dict()
        return basis, report, columns

    # ------------------------------------------------------------------ #
    # Prediction                                                          #
    # ------------------------------------------------------------------ #

    def prepare_columns(self, features: np.ndarray) -> dict[str, np.ndarray]:
        columns = columns_from_matrix(
            np.asarray(features, dtype=np.float64), self.features_name_
        )
        derived = {
            name: (members, "")
            for name, (members, _) in self.derived_variables_.items()
            if all(member in columns for member in members)
        }
        for name, (members, _) in derived.items():
            total = np.zeros(len(next(iter(columns.values()))), dtype=np.float64)
            for member in members:
                total = total + columns[member]
            columns[name] = total
        return columns

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.structure_ is None:
            raise ValueError("Model has not been fit yet.")
        columns = self.prepare_columns(features)
        independent = {
            name: model.predict(columns) for name, model in self.target_models_.items()
        }
        reconstructed = self.structure_.reconstruct(independent, columns)
        bounded = {
            name: clip_to_observed_range(
                np.asarray(values, dtype=np.float64),
                self.target_ranges_.get(name),
                self.config.prediction_range_margin,
            )
            for name, values in reconstructed.items()
        }
        matrix = np.column_stack([bounded[name] for name in self.targets_name_])
        return matrix[:, 0] if matrix.shape[1] == 1 else matrix

    # ------------------------------------------------------------------ #
    # Reporting                                                           #
    # ------------------------------------------------------------------ #

    def get_fit_details(self) -> dict[str, Any]:
        if self.structure_ is None:
            raise ValueError("Model has not been fit yet.")
        return {
            "model": self.name,
            "features": list(self.features_name_),
            "targets": list(self.targets_name_),
            "structure": self.structure_.to_dict(),
            "target_models": [
                model.describe() for model in self.target_models_.values()
            ],
            "config": {
                "use_structure_discovery": self.config.use_structure_discovery,
                "use_dimensional_reduction": self.config.use_dimensional_reduction,
                "refine_power_law": self.config.refine_power_law,
                "log_response": self.config.log_response,
                "feature_space": self.config.feature_space,
            },
        }

    def closed_form(self) -> dict[str, str]:
        """Human-readable formula for every target, when the learner provides one."""
        formulas: dict[str, str] = {}
        for name, model in self.target_models_.items():
            expression = getattr(model.learner, "expression_", None)
            if expression is None:
                formulas[name] = f"<no closed form: {type(model.learner).__name__}>"
                continue
            scale = (
                model.pi_basis.scale_formula() if model.pi_basis is not None else "1"
            )
            if self.config.log_response:
                formulas[name] = f"({scale}) * exp({expression})"
            else:
                formulas[name] = f"({scale}) * ({expression})"
        if self.structure_ is not None:
            for name in self.structure_.reconstruction_order():
                formulas[name] = self.structure_.identities[name].expression
        return formulas
