"""Generate a profile report of the dataframe."""
import copy
from typing import Dict, List, Literal, Optional, TypedDict, Union

import pandas as pd
from pandas_profiling import ProfileReport

from neuralworks.charts import pandas_plotting_backend
from neuralworks.data import get_column_descriptions

# from pandas_profiling.config import Config as PandasProfilingConfig
# from pandas_profiling.config import Settings as PandasProfilingSettings


class _DatasetMetadataBase(TypedDict, total=True):
    """Base (required) dataset metadata dictionary for `pandas_profiling.ProfileReport` report.

    Attributes:
        name: The name of the dataset.

        description: The description of the dataset.
    """
    name: str
    description: str


class DatasetMetadata(_DatasetMetadataBase, total=False):
    """Dataset metadata dictionary for `pandas_profiling.ProfileReport` report.

    Attributes:
        name: The name of the dataset. (required)

        description: The description of the dataset. (required)

    Optional Attributes:
        sample: A sample of the dataset. (optional)
    """
    # name: str
    # description: str
    sample: pd.DataFrame


#
# Default (required) keys for the dataset metadata
DEFAULT_DATASET_METADATA: DatasetMetadata = {
    'name': 'NEURALWORKS CHALLENGE',
    'description': '',  # TODO: Add a description of the dataset
}


def get_default_dataset_metadata() -> DatasetMetadata:
    """Get a deep copy of the default (required) dataset metadata dictionary for `pandas_profiling.ProfileReport`.

    Returns:
        A deep copy of the default (required) dataset metadata.
    """
    return copy.deepcopy(DEFAULT_DATASET_METADATA)


def sample_dataset(
    data: pd.DataFrame,
    **kwargs,
) -> pd.DataFrame:
    """Sample the dataset. By default, randomly selects 10% of the rows and sets the random state for reproducibility.

    Args:
        data: The dataframe to sample from.

        kwargs: Keyword arguments to pass to `pandas.DataFrame.sample()`. Default: `dict(frac=0.1, random_state=42)`

    Returns:
        The sampled dataframe.
    """
    kwargs = {} if kwargs is None else copy.deepcopy(kwargs)
    # if the keys are not in the dictionary, add them with the default values (if any), otherwise do nothing
    _ = kwargs.setdefault('frac', 0.1)  # randomly select 10% (0.1) of the rows
    _ = kwargs.setdefault('random_state', 42)  # set the random state for reproducibility

    return (
        # copy the dataframe
        data.copy(deep=True)
        # sample the dataframe
        .sample(**kwargs)
        # reset the index of the dataframe
        .reset_index(
            drop=False,
            inplace=False,
        )
        # sample of the dataset
    )


class CorrelationCoefficientConfig(TypedDict, total=False):
    """The configuration for the correlation coefficient analysis.

    Attributes:
        calculate: Whether to calculate this coefficient.

        warn_high_correlations: Show warning for correlations higher than the threshold.

        threshold: Warning threshold.
    """
    calculate: bool
    warn_high_correlations: bool
    threshold: float


class CorrelationConfig(TypedDict, total=False):
    """The configuration for the correlation analysis (correlation metrics and thresholds).

    Attributes:
        pearson: The configuration for the `Pearson correlation` analysis.

        spearman: The configuration for the `Spearman correlation`

        kendall: The configuration for the `Kendall correlation` analysis.

        phi_k: The configuration for the `Phi K correlation` analysis.

        cramers: The configuration for the `Cramers correlation` analysis.

        auto: The configuration for the `auto correlation` analysis.
    """
    pearson: CorrelationCoefficientConfig
    spearman: CorrelationCoefficientConfig
    kendall: CorrelationCoefficientConfig
    phi_k: CorrelationCoefficientConfig
    cramers: CorrelationCoefficientConfig
    auto: CorrelationCoefficientConfig


DEFAULT_CORRELATION_CONFIG = CorrelationConfig(
    pearson=CorrelationCoefficientConfig(calculate=True, warn_high_correlations=True, threshold=0.9),
    spearman=CorrelationCoefficientConfig(calculate=True, warn_high_correlations=False, threshold=0.9),
    kendall=CorrelationCoefficientConfig(calculate=True, warn_high_correlations=False, threshold=0.9),
    phi_k=CorrelationCoefficientConfig(calculate=True, warn_high_correlations=False, threshold=0.9),
    cramers=CorrelationCoefficientConfig(calculate=True, warn_high_correlations=True, threshold=0.9),
    auto=CorrelationCoefficientConfig(calculate=True, warn_high_correlations=True, threshold=0.9),
)


class MissingDiagramsConfig(TypedDict, total=False):
    """The configuration for the missing data section and the visualizations it can include.

    Attributes:
        bar: Display a bar chart with counts of missing values for each column.

        matrix: Display a matrix of missing values. Similar to the bar chart, but might provide overview of the co-occurrence of missing values in rows.

        heatmap: Display a heatmap of missing values, that measures nullity correlation (i.e. how strongly the presence or absence of one variable affects the presence of another).
    """
    bar: bool  # pylint: disable=disallowed-name
    matrix: bool
    heatmap: bool


DEFAULT_MISSING_DIAGRAMS_CONFIG = MissingDiagramsConfig(
    bar=True,
    matrix=True,
    heatmap=True,
)


class InteractionsConfig(TypedDict, total=False):
    """The configuration for the interactions analysis.

    Attributes:
        continuous: Generate a 2D scatter plot (or hexagonal binned plot) for all continuous variable pairs.

        targets: When a list of variable names is given, only interactions between these and all other variables are computed.
    """
    continuous: bool
    targets: List[str]


DEFAULT_INTERACTIONS_CONFIG = InteractionsConfig(
    continuous=True,
    targets=[],
)


class VariablesMetadata(TypedDict, total=True):
    """The variables metadata.

    Attributes:
        descriptions: A dictionary of variable descriptions. (required)
    """
    descriptions: Dict[str, str]


DEFAULT_VARIABLES_METADATA: VariablesMetadata = {
    'descriptions': dict(get_column_descriptions()),
}


class StyleConfig(TypedDict, total=False):
    """The configuration for the appearance and style of the report.

    Attributes:
        theme: Select a bootswatch theme. Available options: flatly (dark) and united (orange)

        logo: A base64 encoded logo, to display in the navigation bar.

        primary_color: The primary color to use in the report.

        full_width: By default, the width of the report is fixed. If set to True, the full width of the screen is used.
    """
    theme: Optional[Literal['flatly', 'united']]  # None
    logo: str  # base64 encoded
    primary_color: Union[Literal['#337ab7'], str]
    full_width: bool  # False


DEFAULT_STYLE_CONFIG = StyleConfig(
    theme=None,
    # logo='',  # (base64 encoded)
    primary_color='#337ab7',  # '#337ab7'
    full_width=True,  # False
)


class HtmlConfig(TypedDict, total=False):
    """The configuration for the HTML export.

    Attributes:
        minify_html: If True, the output HTML is minified using the htmlmin package.

        use_local_assets: If True, all assets (stylesheets, scripts, images) are stored locally. If False, a CDN is used for some stylesheets and scripts.

        inline: If True, all assets are contained in the report. If False, then a web export is created, where all assets are stored in the ‘[REPORT_NAME]_assets/’ directory.

        navbar_show: Whether to include a navigation bar in the report

        style: The style configuration for the HTML export.
    """
    minify_html: bool  # True
    use_local_assets: bool  # True
    inline: bool  # True
    navbar_show: bool  # True
    style: StyleConfig


DEFAULT_HTML_CONFIG = HtmlConfig(
    minify_html=True,
    use_local_assets=True,
    inline=True,
    navbar_show=True,
    style=DEFAULT_STYLE_CONFIG,
)


class _BasePandasProfilingReportConfig(TypedDict, total=True):
    """The configuration for the `pandas_profiling.ProfileReport` report.

    Attributes:
        title: Title for the report, shown in the header and title bar. (default: 'Pandas Profiling Report')

        dataset: The dataset metadata.

        show_variable_description: Show the description at each variable (in addition to the overview tab).

        minimal: For large datasets, set minimal=True. (default: False)

        explorative: If True, perform a deeper explorative analysis (default: False)

        sensitive: If True, do not show individual records, only aggregate information (default: False)

        dark_mode: If True, use dark mode (default: False)

        tsmode: If True, use time series mode (default: False)

        sortby: Which variable to sort by. For the analysis work properly (when performing a time series analysis), the dataframe needs to be sorted by entity columns and time, otherwise you can always leverage the `sortby` parameter.

        lazy: If True, do not perform the analysis immediately, but only when the report is generated (default: True)

        pool_size: Number of workers in thread pool. When set to zero, it is set to the number of CPUs available. (`0 = multiprocessing.cpu_count()`). (default: 0)

        progress_bar: If True, show a progress bar when processing (works on jupyter notebook only) (default: True)

        html: Dictionary of additional HTML elements to be added to the report. (default: `{'style': {'full_width': True}}`)
    """
    # https://pandas-profiling.ydata.ai/docs/master/pages/advanced_usage/available_settings.html
    # https://pandas-profiling.ydata.ai/docs/master/pages/use_cases/metadata.html
    title: str
    dataset: DatasetMetadata
    show_variable_description: bool
    minimal: bool
    explorative: bool
    sensitive: bool
    dark_mode: bool
    tsmode: bool
    sortby: Optional[str]
    lazy: bool
    pool_size: int
    progress_bar: bool
    html: HtmlConfig


class PandasProfilingReportConfig(_BasePandasProfilingReportConfig, total=False):
    """The configuration for the `pandas_profiling.ProfileReport` report.

    Attributes:
        title: Title for the report, shown in the header and title bar. (default: 'Pandas Profiling Report')

        dataset: The dataset metadata. (default: `DEFAULT_DATASET_METADATA`)

        show_variable_description: Show the description at each variable (in addition to the overview tab). (default: True)

        minimal: For large datasets, set minimal=True. (default: False)

        explorative: If True, perform a deeper explorative analysis (default: False)

        sensitive: If True, do not show individual records, only aggregate information (default: False)

        dark_mode: If True, use dark mode (default: False)

        tsmode: If True, use time series mode (default: False)

        sortby: Which variable to sort by. For the analysis work properly (when performing a time series analysis), the dataframe needs to be sorted by entity columns and time, otherwise you can always leverage the `sortby` parameter. (default: None)

        lazy: If True, do not perform the analysis immediately, but only when the report is generated (default: True)

        pool_size: Number of workers in thread pool. When set to zero, it is set to the number of CPUs available. (`0 = multiprocessing.cpu_count()`). (default: 0)

        progress_bar: If True, show a progress bar when processing (works on jupyter notebook only) (default: True)

        html: Dictionary of additional HTML elements to be added to the report. (default: `{'style': {'full_width': True}}`)

    Optional Attributes:
        samples:

        correlations: Dictionary of configuration for the correlation analysis (correlation metrics and thresholds).

        missing_diagrams: Dictionary of configuration for the missing data section and the visualizations it can include.

        duplicates:

        interactions: Dictionary of configuration for the interaction analysis.

        variables: Dictionary of variable descriptions.
    """
    # # https://pandas-profiling.ydata.ai/docs/master/pages/advanced_usage/available_settings.html
    # # https://pandas-profiling.ydata.ai/docs/master/pages/use_cases/metadata.html
    # title: str
    # dataset: DatasetMetadata
    # show_variable_description: bool
    # minimal: bool
    # explorative: bool
    # sensitive: bool
    # dark_mode: bool
    # tsmode: bool
    # sortby: Optional[str]
    # lazy: bool
    # pool_size: int
    # progress_bar: bool
    # html: HtmlConfig
    samples: Optional[dict]
    correlations: Optional[CorrelationConfig]
    missing_diagrams: Optional[MissingDiagramsConfig]
    duplicates: Optional[dict]
    interactions: Optional[InteractionsConfig]
    variables: Optional[VariablesMetadata]


PROFILE_KWARGS: PandasProfilingReportConfig = {
    'title': 'Pandas Profiling Report',
    'dataset': DEFAULT_DATASET_METADATA,
    'show_variable_description': True,
    'minimal': False,
    'explorative': False,
    'sensitive': False,
    'dark_mode': False,
    'tsmode': False,  # Time Series Configuration
    'sortby': None,  # Time Series Configuration
    'lazy': True,
    'pool_size': 0,  # 0 means that all CPUs are used (0 = multiprocessing.cpu_count())
    'progress_bar': True,
    'html': {
        'minify_html': True,  # True
        'use_local_assets': True,  # True
        'inline': True,  # True
        'navbar_show': True,  # True
        'style': {
            # 'theme': None,
            # 'logo': '',
            # 'primary_color': '#337ab7',
            # By default, the width of the report is fixed. If set to True, the full width of the screen is used.
            'full_width': True,
        },
    },
    'variables': {
        'descriptions': dict(get_column_descriptions()),
    },
    # #
    # 'vars': {
    #     'timeseries': {
    #         # 'active': False,  # bool
    #         # 'sortby': None,  # Optional[str]
    #         # 'autocorrelation': 0.7,  # float
    #         # 'lags': [1, 7, 12, 24, 30],  # List[int]
    #         # 'significance': 0.05,  # float
    #         # 'pacf_acf_lag': 100,  # int
    #     },
    # },
    # #
    # #
    # # NOTE: To disable samples, correlations, missing diagrams and duplicates at once, set them to None
    # #
    # # # control whether the dataset preview is shown.
    # # 'samples': None,
    # #
    # # # control whether correlation computations are executed.
    # # 'correlations': None,
    # #
    # 'correlations': {
    #     'pearson': {
    #         'calculate': True,
    #         'warn_high_correlations': True,
    #         'threshold': 0.9,
    #     },
    #     'spearman': {
    #         'calculate': True,
    #         'warn_high_correlations': False,
    #         'threshold': 0.9,
    #     },
    #     'kendall': {
    #         'calculate': True,
    #         'warn_high_correlations': False,
    #         'threshold': 0.9,
    #     },
    #     'phi_k': {
    #         'calculate': True,
    #         'warn_high_correlations': False,
    #         'threshold': 0.9,
    #     },
    #     'cramers': {
    #         'calculate': True,
    #         'warn_high_correlations': True,
    #         'threshold': 0.9,
    #     },
    #     'auto': {
    #         'calculate': True,
    #         'warn_high_correlations': True,
    #         'threshold': 0.9,
    #     },
    # },
    # #
    # # # control whether missing value analysis is executed.
    # # 'missing_diagrams': None,
    # 'missing_diagrams': {
    #     'bar': True,
    #     'matrix': True,
    #     'heatmap': True,
    # },
    # #
    # # # control whether duplicate rows are previewed.
    # # 'duplicates': None,
    # #
    # # # control whether interactions are computed.
    # # 'interactions': None,
    # 'interactions': {
    #     'continuous': True,
    #     'targets': [],
    # }
    # #
}


def get_default_profile_kwargs() -> PandasProfilingReportConfig:
    """Get a deep copy of the default `pandas_profiling.ProfileReport` report kwargs (useful for PyCaret)."""
    return copy.deepcopy(PROFILE_KWARGS)


# TODO: Check if 'tsmode' (time series mode) parameter is useful
def generate_profile_report(
    data: pd.DataFrame,
    minimal: bool = False,  # For large datasets, set minimal=True. (default: False)
    explorative: bool = False,  # If True, perform a deeper explorative analysis (default: False)
    # Time Series Configuration
    tsmode: bool = False,  # If True, use time series mode (default: False),
    sortby: Optional[str] = None,
    *,
    # Sampling Configuration
    sample_the_dataset: bool = False,
    sample_kwargs: Optional[Dict[str, object]] = None,
) -> ProfileReport:
    """Generate a profile report of the dataframe (using `pandas_profiling.ProfileReport`).

    NOTE: Use it with `with pandas_plotting_backend(backend='matplotlib'):`

    Args:
        data: The dataframe.

        minimal: For large datasets, set minimal=True. (default: False)

        explorative: If True, perform a deeper explorative analysis (default: False)

        tsmode: If True, use time series mode (default: False)

        sortby: Which variable to sort by. For the analysis work properly (when performing a time series analysis), the dataframe needs to be sorted by entity columns and time, otherwise you can always leverage the `sortby` parameter.

        sample_the_dataset: If True, generate the report using a sample of the dataset. (default: False)

        sample_kwargs: Keyword arguments to pass to `pandas.DataFrame.sample()` (if `sample_the_dataset` is True). Default: `dict(frac=0.1, random_state=42)`

    Returns:
        The profile report of the dataframe.
    """
    dataset_metadata: DatasetMetadata = copy.deepcopy(DEFAULT_DATASET_METADATA)
    if sample_the_dataset:
        # copy the sample kwargs (if provided) to avoid modifying the original dictionary
        sample_kwargs = {} if sample_kwargs is None else copy.deepcopy(sample_kwargs)
        # set default values for the sample kwargs (if not provided)
        _ = sample_kwargs.setdefault('frac', 0.1)  # randomly select 10% (0.1) of the rows
        _ = sample_kwargs.setdefault('random_state', 42)  # set the random state for reproducibility
        # sample the dataset and add it to the dataset metadata dictionary
        _ = dataset_metadata.setdefault('sample', sample_dataset(data=data, **sample_kwargs))

    # For the analysis work properly (when performing a time series analysis), the dataframe needs to be sorted by
    # entity columns and time, otherwise you can always leverage the sortby parameter.
    # Ref: https://towardsdatascience.com/how-to-do-an-eda-for-time-series-cbb92b3b1913

    # get a deep copy the profile kwargs
    profile_kwargs: PandasProfilingReportConfig = get_default_profile_kwargs()

    # update the profile kwargs with the provided arguments
    profile_kwargs['dataset'] = dataset_metadata
    profile_kwargs['minimal'] = minimal
    profile_kwargs['explorative'] = explorative
    profile_kwargs['tsmode'] = tsmode
    profile_kwargs['sortby'] = sortby

    # NOTE: To disable samples, correlations, missing diagrams and duplicates at once, set them to None
    # profile_kwargs['samples'] = None
    # profile_kwargs['correlations'] = None
    # profile_kwargs['missing_diagrams'] = None
    # profile_kwargs['duplicates'] = None
    # profile_kwargs['interactions'] = None

    # NOTE: 'plotly' plotting backend is not compatible with pandas-profiling
    with pandas_plotting_backend(backend='matplotlib'):
        # pandas profiling report
        report = ProfileReport(df=data.copy(deep=True), **profile_kwargs)

        # # https://pandas-profiling.ydata.ai/docs/master/pages/advanced_usage/multiple_runs.html
        # report.invalidate_cache(
        #     subset=None,  # None (default) to invalidate all caches
        #   )

        # https://pandas-profiling.ydata.ai/docs/master/pages/advanced_usage/corr_mat_access.html
        # https://stackoverflow.com/questions/64621116/how-can-i-get-the-numbers-for-the-correlation-matrix-from-pandas-profiling

        # # Listing available correlation
        # correlations = report.description_set["correlations"]
        #  print(correlations.keys())

        # Accessing the values of the Pearson correlation
        #  # DataFrame where row and column names are the names of the original columns
        #   pearson_df = correlations["pearson"]

        # # Actual correlation values
        # pearson_mat = pearson_df.values
        # print(pearson_mat)

        return report
