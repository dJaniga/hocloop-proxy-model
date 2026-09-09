import argparse
import logging
from pathlib import Path

from toolbox import ExperimentSettings, pipeline

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler("hocloop-proxy-model.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hocloop structured proxy model: symbolic regression with "
        "target-structure discovery and dimensional reduction.",
    )
    parser.add_argument(
        "--features-file",
        type=Path,
        help="Path to the feature file (with header)",
        required=True,
    )
    parser.add_argument(
        "--targets-file",
        type=Path,
        help="Path to the targets file (with header)",
        required=True,
    )
    parser.add_argument(
        "--output-path", type=Path, help="Directory for all artefacts", required=True
    )
    parser.add_argument(
        "--units-file",
        type=Path,
        default=None,
        help='JSON with {"features": {...}, "targets": {...}} mapping each column '
        'to a flat unit string such as "W/m/K". Defaults to the built-in HOCLOOP units.',
    )

    parser.add_argument(
        "--learners",
        nargs="+",
        default=[
            "power_law_ols",
            "symbolic_deap",
            "symbolic_residual",
            "symbolic_pysr",
            "polynomial_ridge",
            "gradient_boosting",
            "random_forest",
            "neural_network",
        ],
        help="Learners to compare in the benchmark table.",
    )
    parser.add_argument(
        "--primary-learner",
        default="symbolic_residual",
        help="Learner used for the ablation, learning curve, conformal intervals "
        "and the reported closed form.",
    )
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=1)
    parser.add_argument(
        "--shell-fraction",
        type=float,
        default=0.25,
        help="Fraction of each feature range treated as the extrapolation shell.",
    )
    parser.add_argument("--conformal-alpha", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--generations", type=int, default=120, help="GP generations.")
    parser.add_argument("--population-size", type=int, default=400)
    parser.add_argument("--max-tree-height", type=int, default=6)
    parser.add_argument("--n-islands", type=int, default=4)
    parser.add_argument(
        "--parallel-islands",
        action="store_true",
        help="Evolve islands in worker processes.",
    )

    parser.add_argument(
        "--tune",
        action="store_true",
        help="Search hyperparameters inside every training fold. The search runs "
        "once per fit, so reported scores are nested cross-validation, and the "
        "cost is n_trials times the number of fits.",
    )
    parser.add_argument("--n-trials", type=int, default=25, help="Optuna trials per fit.")
    parser.add_argument("--inner-splits", type=int, default=3, help="Inner folds per trial.")
    parser.add_argument(
        "--tuning-metric",
        default="r2_log_score",
        help="Objective for the search. Log-scale R-squared by default, because "
        "raw R-squared on these targets is decided by a few extreme points.",
    )
    parser.add_argument(
        "--tune-pipeline",
        action="store_true",
        help="Also search the feature space and the power-law refinement switch.",
    )
    parser.add_argument(
        "--tune-learners",
        nargs="+",
        default=[],
        help="Learners to tune. Default: every learner that has a search space.",
    )

    parser.add_argument("--skip-benchmarks", action="store_true")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--skip-learning-curve", action="store_true")
    parser.add_argument("--skip-conformal", action="store_true")
    parser.add_argument("--skip-design-rules", action="store_true")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Small GP budget and a single CV repeat, for a smoke run.",
    )
    parser.add_argument("--verbose", action="store_true")

    return parser


def settings_from_args(args: argparse.Namespace) -> ExperimentSettings:
    symbolic_overrides = {
        "generations": 25 if args.quick else args.generations,
        "population_size": 150 if args.quick else args.population_size,
        "max_tree_height": args.max_tree_height,
        "n_islands": 2 if args.quick else args.n_islands,
        "parallel_islands": args.parallel_islands,
        "seed": args.seed,
    }
    return ExperimentSettings(
        learners=tuple(args.learners),
        primary_learner=args.primary_learner,
        cv_splits=3 if args.quick else args.cv_splits,
        cv_repeats=1 if args.quick else args.cv_repeats,
        shell_fraction=args.shell_fraction,
        conformal_alpha=args.conformal_alpha,
        conformal_splits=3 if args.quick else 5,
        learning_curve_sizes=(200, 800, 3200) if args.quick else (100, 200, 400, 800, 1600, 3200),
        learning_curve_repeats=1 if args.quick else 2,
        run_benchmarks=not args.skip_benchmarks,
        run_ablation=not args.skip_ablation,
        run_learning_curve=not args.skip_learning_curve,
        run_conformal=not args.skip_conformal,
        run_design_rules=not args.skip_design_rules,
        seed=args.seed,
        symbolic_overrides=symbolic_overrides,
        tune=args.tune,
        n_trials=5 if args.quick else args.n_trials,
        inner_splits=2 if args.quick else args.inner_splits,
        tuning_metric=args.tuning_metric,
        tune_pipeline=args.tune_pipeline,
        tune_learners=tuple(args.tune_learners),
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(args.verbose)

    pipeline(
        args.features_file,
        args.targets_file,
        args.output_path,
        units_file=args.units_file,
        settings=settings_from_args(args),
    )


if __name__ == "__main__":
    main()
