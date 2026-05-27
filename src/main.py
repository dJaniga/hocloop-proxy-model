import argparse
import logging
from pathlib import Path

from toolbox import pipeline

logger = logging.getLogger(__name__)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler("hocloop-proxy-model.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hocloop Proxy Model",
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

    return parser


def main():
    setup_logging()
    parser = build_parser()
    args = parser.parse_args()

    pipeline(args.features_file, args.targets_file)


if __name__ == "__main__":
    main()
