import logging
from pathlib import Path

import pandas as pd

from modeling import ModelWrapper
from modeling.symbolic import SymbolicRegressor

logger = logging.getLogger(__name__)


def load_data(file: Path) -> pd.DataFrame:
    logger.info(f"Loading data from {file}")
    df = pd.read_csv(file)
    return df


def pipeline(feature_file_path: Path, target_file_path: Path, output_path: Path) -> None:
    logger.info("Starting symbolic regression pipeline")

    feature_df = load_data(feature_file_path)
    target_df = load_data(target_file_path)

    feature_names = tuple(feature_df.columns)
    target_names = tuple(target_df.columns)
    idx = feature_df.index

    features = feature_df.to_numpy()
    targets = target_df.to_numpy()

    regressor = SymbolicRegressor()
    regressor_wrapper = ModelWrapper(model=regressor, output_path=output_path, n_jobs=2)
    regressor_wrapper.fit(idx, features, targets, feature_names, target_names, optimize_hyperparameters=False)

    fit_details = regressor.get_fit_details()
    logger.info(fit_details)
