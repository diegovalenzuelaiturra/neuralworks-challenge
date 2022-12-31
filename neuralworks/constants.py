"""Constants for the project."""
from pathlib import Path
from typing import NamedTuple

__all__ = [
    # common
    'get_project_root_dir_path',
    # data
    'get_data_dir_path',
    'get_raw_data_dir_path',
    'get_processed_data_dir_path',
    # logs
    'get_path_to_logs_dir_path',
    # reports
    'get_reports_dir_path',
    # custom
    'get_data_file_path',
    'get_synthetic_features_file_path',
    # plotly
    'get_custom_plotly_figure_size',
    # assets
    'get_assets_dir_path',
]

# Common
ROOT_DIR: Path = Path(__file__).parent.parent


def get_project_root_dir_path() -> Path:
    """Return the path to the project root directory."""
    return ROOT_DIR


# Data
DATA_DIR_PATH: Path = ROOT_DIR / 'data'
RAW_DATA_DIR_PATH: Path = ROOT_DIR / 'data' / 'raw'
PROCESSED_DATA_DIR_PATH: Path = ROOT_DIR / 'data' / 'processed'


def get_data_dir_path() -> Path:
    """Return the path to the data directory."""
    return DATA_DIR_PATH


def get_raw_data_dir_path() -> Path:
    """Return the path to the raw data directory."""
    return RAW_DATA_DIR_PATH


def get_processed_data_dir_path() -> Path:
    """Return the path to the processed data directory."""
    return PROCESSED_DATA_DIR_PATH


# Logs
PATH_TO_LOGS_DIR_PATH: Path = ROOT_DIR / 'logs'


def get_path_to_logs_dir_path() -> Path:
    """Return the path to the logs directory."""
    return PATH_TO_LOGS_DIR_PATH


# Reports
PATH_TO_REPORTS_DIR_PATH: Path = ROOT_DIR / 'reports'


def get_reports_dir_path() -> Path:
    """Return the path to the reports directory."""
    return PATH_TO_REPORTS_DIR_PATH


# Custom

DATASET_FILE_NAME: str = 'dataset_SCL.csv'
SYNTHETIC_FEATURES_FILE_NAME: str = 'synthetic_features.csv'

DATA_FILE_PATH: Path = ROOT_DIR / 'data' / 'raw' / DATASET_FILE_NAME
SYNTHETIC_DATA_FILE_PATH: Path = ROOT_DIR / 'data' / 'processed' / SYNTHETIC_FEATURES_FILE_NAME


def get_data_file_path() -> Path:
    """Return the path to the dataset file."""
    return DATA_FILE_PATH


def get_synthetic_features_file_path() -> Path:
    """Return the path to the synthetic features file."""
    return SYNTHETIC_DATA_FILE_PATH


# Plotly


class PlotlyFigureSize(NamedTuple):
    """Named tuple for `Plotly` figure size `(width, height)`.

    Attributes:
        width (float): width.
        height (float): height.

    Examples:
        Values roughly equivalent to `matplotlib` figure size `(16, 9)` are the following:

        >>> width  = 16 * 100 * 0.835  # 1336.0
        >>> height =  9 * 100 * 0.835  # 751.5
        >>> PlotlyFigureSize(width=width, height=height)
        PlotlyFigureSize(width=1336.0, height=751.5)
    """
    width: float
    height: float


# Roughly equivalent to `matplotlib` figure size `(16, 9)` (width, height) i.e. (1336.0, 751.5) in pixels
CUSTOM_PLOTLY_FIGURE_SIZE: PlotlyFigureSize = PlotlyFigureSize(
    width=16 * 100 * 0.835,
    height=9 * 100 * 0.835,
)


def get_custom_plotly_figure_size() -> PlotlyFigureSize:
    """Return the `Plotly` figure size `(width, height)`."""
    return CUSTOM_PLOTLY_FIGURE_SIZE


# Assets
PATH_TO_ASSETS_DIR_PATH: Path = ROOT_DIR / 'assets'


def get_assets_dir_path() -> Path:
    """Return the path to the assets directory."""
    return PATH_TO_ASSETS_DIR_PATH
