"""Code for loading data."""
from __future__ import annotations

from collections import OrderedDict
import copy
import logging
from pathlib import Path
from typing import Callable, Dict, Literal, Optional, Tuple, TYPE_CHECKING, Union

from IPython.display import display
from IPython.display import Markdown
import pandas as pd
from pandas.io.formats.style import Styler
from scipy import stats

from neuralworks.constants import get_data_file_path
from neuralworks.constants import get_synthetic_features_file_path

if TYPE_CHECKING:
    FilePath = Union[str, Path]

logger: logging.Logger = logging.getLogger(__name__)

# default styles for the tables in the notebook
TABLE_STYLES = [
    # title
    dict(
        selector='caption',
        props=[
            ('text-align', 'center'),
            ('font-size', '150%'),
            ('font-weight', 'bold'),
            ('caption-side', 'top'),
            ('margin-bottom', '1.5em'),
        ],
    ),
    # headers
    dict(
        selector='th',
        props=[
            ('text-align', 'center'),
            ('font-size', '125%'),
            ('font-weight', 'bold'),
        ],
    ),
    # cells
    dict(
        selector='td',
        props=[
            ('text-align', 'left'),
            ('font-size', '100%'),
        ],
    ),
]

COLUMN_DESCRIPTIONS = OrderedDict(
    (
        # BASIC INFORMATION
        ('Fecha-I', 'Fecha y hora programada del vuelo.'),
        ('Vlo-I', 'Número de vuelo programado.'),
        ('Ori-I', 'Código de ciudad de origen programado.'),
        ('Des-I', 'Código de ciudad de destino programado.'),
        ('Emp-I', 'Código aerolínea de vuelo programado.'),
        ('Fecha-O', 'Fecha y hora de operación del vuelo.'),
        ('Vlo-O', 'Número de vuelo de operación del vuelo.'),
        ('Ori-O', 'Código de ciudad de origen de operación'),
        ('Des-O', 'Código de ciudad de destino de operación.'),
        ('Emp-O', 'Código aerolínea de vuelo operado.'),
        ('DIA', 'Día del mes de operación del vuelo.'),
        ('MES', 'Número de mes de operación del vuelo.'),
        ('AÑO', 'Año de operación del vuelo.'),
        ('DIANOM', 'Día de la semana de operación del vuelo.'),
        ('TIPOVUELO', 'Tipo de vuelo, `I` =Internacional,`N` =Nacional.'),
        ('OPERA', 'Nombre de aerolínea que opera.'),
        ('SIGLAORI', 'Nombre ciudad origen.'),
        ('SIGLADES', 'Nombre ciudad destino.'),
        # CUSTOM (SYNTHETIC) FEATURES
        # TODO: add descriptions for the synthetic features
    ),)


def get_column_descriptions() -> OrderedDict[str, str]:
    """Return a dictionary of variable names and their descriptions."""
    # copy the dictionary to avoid modifying the original
    return copy.deepcopy(COLUMN_DESCRIPTIONS)


def get_column_descriptions_df(sort: bool = False) -> pd.DataFrame:
    """Return a dataframe with the variable names and their descriptions.

    Args:
        sort (bool, optional): Sort the dataframe by variable name. Defaults to False.
    """
    return (
        # create a dataframe with the variable names and their descriptions
        pd.DataFrame.from_dict(
            data=get_column_descriptions(),  # get the column descriptions OrderedDict
            orient='index',
            columns=['Descripción'],
        )
        # set the index name to 'name'
        .rename_axis(index='Variable')
        # sort the dataframe by the variable names if sorted is True
        .pipe(lambda df: df.sort_index() if not df.empty and sort else df)
        # return the dataframe
    )


# NOTE: categories use less memory than integers ('int64') or strings ('object') and are faster to process

# dtypes for the original data
DATASET_DTYPES = {
    # ---------------------------------------------------------------------------------------------------------------
    # 'Fecha-I': 'datetime64[ns]',  # Use parse_dates instead
    'Vlo-I':
        'category',
    'Ori-I':
        'category',
    'Des-I':
        'category',
    'Emp-I':
        'category',
    # ---------------------------------------------------------------------------------------------------------------
    # 'Fecha-O': 'datetime64[ns]',  # Use parse_dates instead
    'Vlo-O':
        'category',
    'Ori-O':
        'category',
    'Des-O':
        'category',
    'Emp-O':
        'category',
    # ---------------------------------------------------------------------------------------------------------------
    # 'DIA': 'category',  # (1, 2, ..., 31)
    'DIA':
        pd.CategoricalDtype(
            categories=[str(d) for d in range(1, 32)],
            ordered=True,
        ),
    # 'MES': 'category',  # (1, 2, ..., 12)
    'MES':
        pd.CategoricalDtype(
            categories=[str(m) for m in range(1, 13)],
            ordered=True,
        ),
    # 'AÑO': 'category',  # (2017, 2018)
    'AÑO':
        'category',
    # 'DIANOM': 'category',  # (Lunes, Martes, ..., Domingo)
    'DIANOM':
        pd.CategoricalDtype(
            categories=['Lunes', 'Martes', 'Miercoles', 'Jueves', 'Viernes', 'Sabado', 'Domingo'],
            ordered=True,
        ),
    # ---------------------------------------------------------------------------------------------------------------
    # 'TIPOVUELO': 'category',  # (N, I)  # i.e. Nacional, Internacional
    'TIPOVUELO':
        pd.CategoricalDtype(
            categories=['N', 'I'],
            ordered=False,
        ),
    # ---------------------------------------------------------------------------------------------------------------
    'OPERA':
        'category',
    # ---------------------------------------------------------------------------------------------------------------
    'SIGLAORI':
        'category',
    'SIGLADES':
        'category',
    # ---------------------------------------------------------------------------------------------------------------
}

# custom type for the type of synthetic features to generate (used in generate_synthetic_features)
SYNTHETIC_FEATURES_DTYPES = {
    'temporada_alta': pd.CategoricalDtype(
        categories=[0, 1],
        ordered=False,
    ),
    'dif_min': 'float64',
    'atraso_15': pd.CategoricalDtype(
        categories=[0, 1],
        ordered=False,
    ),
    'periodo_dia': pd.CategoricalDtype(
        categories=['mañana', 'tarde', 'noche'],
        ordered=True,
    ),
}

# low_memory bool, default True
#   Internally process the file in chunks, resulting in lower memory use while parsing, but possibly mixed type inference.
#   To ensure no mixed types either set False, or specify the type with the dtype parameter.
# Note that the entire file is read into a single DataFrame regardless,
# use the chunksize or iterator parameter to return the data in chunks.
# (Only valid with C parser).

# memory_map bool, default False
#   If a filepath is provided for filepath_or_buffer, map the file object directly onto memory and access the data directly from there.
#   Using this option can improve performance because there is no longer any I/O overhead.

# def load_csv(filepath_or_buffer: 'FilePath') -> pd.DataFrame:
#     """Load the data from the CSV file.
#
#     Args:
#         filepath_or_buffer: The path to the CSV file or a file-like object.
#
#     Returns:
#         A dataframe containing the data from the CSV file.
#     """
#     # ASCII text, with CRLF line terminators and uses | as a field separator
#     # where the decimal separator is a comma
#     logger.info('Loading data from %s', filepath_or_buffer)
#     return pd.read_csv(
#         filepath_or_buffer,
#         # low_memory=False,  # Set to False to ensure no mixed types
#         # memory_map=True,  # Set to True to map the file object directly onto memory (improves performance)
#         dtype=DATASET_DTYPES,
#         parse_dates=['Fecha-I', 'Fecha-O'],
#         infer_datetime_format=True,
#         date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'),
#     )


def load_data() -> pd.DataFrame:
    """Load data from file.

    Returns:
        DataFrame with data.
    """
    path = get_data_file_path()

    logger.info('Loading data from %s', path)
    return pd.read_csv(
        path,
        dtype=DATASET_DTYPES,  # TODO: CHECK THIS WITH THE REPORT
        parse_dates=['Fecha-I', 'Fecha-O'],
        infer_datetime_format=True,
        date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'),
        # CUSTOM
        low_memory=False,  # Set to False to ensure no mixed types
        memory_map=True,  # Set to True to map the file object directly onto memory (improves performance)
    )


def difference_in_minutes(
    x: pd.Series[pd.Timestamp],
    y: pd.Series[pd.Timestamp],
) -> pd.Series[float]:
    """Difference (x - y) in minutes between two Series of Timestamps."""
    return (x - y) / pd.Timedelta(minutes=1)


def delay_15_minutes(x: pd.Series[float]) -> pd.Series:
    """Delayed more than 15 minutes (1 = yes, 0 = no)."""
    return (x > 15).astype(int)


def period_of_the_day(x: pd.Series[pd.Timestamp]) -> pd.Series:
    """Period of the day ('mañana' = morning, 'tarde' = afternoon, 'noche' = night)."""
    # morning   (between 05:00 and 11:59) (05 <= hour < 12)
    # afternoon (between 12:00 and 18:59) (12 <= hour < 19)
    # night     (between 19:00 and 04:59) (19 <= hour < 05)
    return pd.cut(
        x.dt.hour,  # only use the hour part of the datetime object (ignore minutes and seconds)
        bins=[0, 5, 12, 19, 24],  # 0:00 to 4:59, 5:00 to 11:59, 12:00 to 18:59, 19:00 to 23:59
        labels=['noche', 'mañana', 'tarde', 'noche'],
        right=False,  # default is True, but we want the interval to include the right endpoint
        include_lowest=True,  # default is False, but we want the interval to include the left endpoint
        ordered=False,
    )


def high_season(x: pd.Series[pd.Timestamp]) -> pd.Series:
    """High season (1 = high season, 0 = low season)."""

    def func(x: pd.Timestamp) -> int:
        """Return 1 if the date is in the high season, 0 otherwise."""
        # sourcery skip: assign-if-exp, reintroduce-else

        #   between '15-Dic' (12/15) and '3-Mar' (03/03) (inclusive)
        #   between '15-Jul' (07/15) and '31-Jul' (07/31) (inclusive)
        #   between '11-Sep' (09/11) and '30-Sep' (09/30) (inclusive)
        month = x.month
        day = x.day

        # between '15-Dic' (12/15) and '3-Mar' (03/03) (inclusive)
        if (month == 12 and day >= 15) or (month == 1) or (month == 2) or (month == 3 and day <= 3):
            return 1
        # between '15-Jul' (07/15) and '31-Jul' (07/31) (inclusive)
        if month == 7 and 15 <= day <= 31:  # (month == 7 and day >= 15)
            return 1
        # between '11-Sep' (09/11) and '30-Sep' (09/30) (inclusive)
        if month == 9 and 11 <= day <= 30:  # (month == 9 and day >= 11)
            return 1
        # otherwise (low season)
        return 0

    return x.apply(func)


def create_synthetic_features(df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Create synthetic features from original data.

    Args:
        df: DataFrame with original data. If None, load data from file. Default: None.

    Returns:
        DataFrame with synthetic features.
    """
    if df is None:
        # load data if not provided
        df = load_data()
        # only keep the columns we need (to save memory)
        df = df[['Fecha-I', 'Fecha-O']]
    else:
        # only keep the columns we need (to save memory) and create a copy to avoid modifying the original dataframe.
        df = df[['Fecha-I', 'Fecha-O']].copy(deep=True)

    # difference between actual departure time and scheduled departure time, in minutes (float64).
    df['dif_min'] = difference_in_minutes(
        x=df['Fecha-O'],
        y=df['Fecha-I'],
    ).astype(SYNTHETIC_FEATURES_DTYPES['dif_min'])

    # delay of more than 15 minutes (1 = yes, 0 = no) (category).
    df['atraso_15'] = delay_15_minutes(x=df['dif_min']).astype(SYNTHETIC_FEATURES_DTYPES['atraso_15'])

    # period of the day ('mañana' = morning, 'tarde' = afternoon, 'noche' = night) (category).
    #   morning   (between 05:00 and 11:59) (05 <= hour < 12)
    #   afternoon (between 12:00 and 18:59) (12 <= hour < 19)
    #   night     (between 19:00 and 04:59) (19 <= hour < 05)
    df['periodo_dia'] = period_of_the_day(x=df['Fecha-I']).astype(SYNTHETIC_FEATURES_DTYPES['periodo_dia'])

    # high season (1 = yes, 0 = no) in which the flight was scheduled to depart.
    #   between '15-Dic' (12/15) and '3-Mar' (03/03) (inclusive)
    #   between '15-Jul' (07/15) and '31-Jul' (07/31) (inclusive)
    #   between '11-Sep' (09/11) and '30-Sep' (09/30) (inclusive)
    df['temporada_alta'] = high_season(x=df['Fecha-I']).astype(SYNTHETIC_FEATURES_DTYPES['temporada_alta'])

    # only keep the columns we are interested in (drop the rest) and return the resulting DataFrame
    return df[[
        # 'Fecha-I', # we don't need this column anymore
        # 'Fecha-O', # we don't need these column anymore
        'dif_min',
        'atraso_15',
        'periodo_dia',
        'temporada_alta',
    ]]


def remove_synthetic_features_file() -> None:
    """Remove synthetic features file (if it exists)."""
    path = get_synthetic_features_file_path()

    if path.exists():
        logger.info('Removing file %s...', path)
        path.unlink()

    return None


def save_synthetic_features_to_file(
    df: pd.DataFrame,
    overwrite: bool = False,
) -> None:
    """Save synthetic features to file.

    Args:
        df: DataFrame with synthetic features.
        overwrite: Whether to overwrite the file if it already exists. Default: False.

    Raises:
        FileExistsError: if file already exists.
    """
    path = get_synthetic_features_file_path()

    if path.exists():
        if overwrite:
            logger.warning('File already exists: %s', path)
            logger.info('Removing file %s...', path)
            path.unlink()
        else:
            logger.error('File already exists: %s', path)
            raise FileExistsError(f'File already exists: {path}')

    logger.info('Saving synthetic features to file %s...', path)
    df.to_csv(
        path,
        index=False,  # do not save index column (0, 1, 2, ...) to file (it is not needed) (default: True)
    )
    logger.info('Synthetic features saved to file %s.', path)

    return None


def load_synthetic_features() -> pd.DataFrame:
    """Load synthetic features from file.

    Returns:
        DataFrame with synthetic features.
    """
    path = get_synthetic_features_file_path()
    logger.info('Loading synthetic features from file %s...', path)
    return pd.read_csv(path, dtype=SYNTHETIC_FEATURES_DTYPES)


def __period_of_the_day(x: pd.Series[pd.Timestamp]) -> pd.Series:
    """Period of the day ('mañana' = morning, 'tarde' = afternoon, 'noche' = night) as categorical data."""

    def func(x: pd.Timestamp) -> Literal['mañana', 'tarde', 'noche']:
        """Return period of the day as string."""
        # sourcery skip: assign-if-exp, reintroduce-else

        # morning   (between 05:00 and 11:59) (05 <= hour < 12)
        # afternoon (between 12:00 and 18:59) (12 <= hour < 19)
        # night     (between 19:00 and 04:59) (19 <= hour < 05)
        hour = x.hour
        if 5 <= hour < 12:
            return 'mañana'
        if 12 <= hour < 19:
            return 'tarde'
        # if 19 <= hour < 24 or 0 <= hour < 5:
        return 'noche'

    return x.apply(func).astype(
        # cast to category to save memory (instead of using strings)
        pd.CategoricalDtype(
            categories=[
                'mañana',
                'tarde',
                'noche',
            ],
            ordered=False,
        ))


def __create_synthetic_features(df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    # load data from file if not provided, otherwise create a copy of the provided DataFrame
    df = load_data() if df is None else df.copy(deep=True)

    # transform `Fecha-I` and `Fecha-O` to datetime format (if not already done).
    df['Fecha-I'] = pd.to_datetime(df['Fecha-I'])  # TimestampSeries  # already done in load_data()
    df['Fecha-O'] = pd.to_datetime(df['Fecha-O'])  # TimestampSeries  # already done in load_data()

    # difference between actual departure time and scheduled departure time, in minutes (float64).
    df['dif_min'] = (df['Fecha-O'] - df['Fecha-I']).dt.total_seconds() / 60.0
    # df['dif_min'] = (df['Fecha-O'] - df['Fecha-I']) / pd.Timedelta(minutes=1)

    # delay of more than 15 minutes (1 = yes, 0 = no) (int64/category).
    df['atraso_15'] = (df['dif_min'] > 15).astype(int).astype('category')

    # period of the day (morning, afternoon, night) as categorical variable.
    #   morning   (between 05:00 and 11:59) (05 <= hour < 12)
    #   afternoon (between 12:00 and 18:59) (12 <= hour < 19)
    #   night     (between 19:00 and 04:59) (19 <= hour < 05)
    # df['periodo_dia'] = pd.cut(
    #     df['Fecha-I'].dt.hour,  # only use the hour part of the datetime object (ignore minutes and seconds)
    #     bins=[0, 5, 12, 19, 24],  # 0:00 to 4:59, 5:00 to 11:59, 12:00 to 18:59, 19:00 to 23:59
    #     labels=['noche', 'mañana', 'tarde', 'noche'],
    #     right=False,  # default is True, but we want the interval to include the right endpoint
    #     include_lowest=True,  # default is False, but we want the interval to include the left endpoint
    #     ordered=False,
    # ).astype('category')

    # # same as above but using apply() function
    df['periodo_dia'] = df[
        'Fecha-I'].dt.hour.apply(  # only use the hour part of the datetime object (ignore minutes and seconds)
            lambda x:
            # morning   (between 05:00 and 11:59)
            'mañana' if 5 <= x <= 11 else
            # afternoon (between 12:00 and 18:59)
            'tarde' if 12 <= x <= 18 else
            # night     (between 19:00 and 04:59)
            'noche')

    # transform `periodo_dia` to category type (categories use less memory than strings ('object') for this column)
    df['periodo_dia'] = df['periodo_dia'].astype(
        # 'category'
        pd.CategoricalDtype(
            categories=[
                'mañana',
                'tarde',
                'noche',
            ],
            ordered=False,
        ))

    # high season (1 = yes, 0 = no) in which the flight was scheduled to depart.
    df['temporada_alta'] = df['Fecha-I'].apply(  # use the whole datetime object (including hour, minutes and seconds)
        lambda x: 1 if (  # x is a pd.Timestamp object
            # between '15-Dic' and '3-Mar'
            (x.month == 12 and x.day >= 15) or (x.month == 1) or (x.month == 2) or (x.month == 3 and x.day <= 3) or
            # between '15-Jul' and '31-Jul'
            (x.month == 7 and x.day >= 15) or
            # between '11-Sep' and '30-Sep'
            (x.month == 9 and x.day >= 11))
        # otherwise
        else 0)

    # transform high season as category instead of integer (category uses less memory than integer)
    df['temporada_alta'] = df['temporada_alta'].astype('category')

    # only keep the columns we are interested in (drop the rest) and return the resulting DataFrame
    return df[[
        # 'Fecha-I',
        # 'Fecha-O',
        #
        # difference between actual departure time and scheduled departure time, in minutes (float64).
        'dif_min',
        # delay of more than 15 minutes (1 = yes, 0 = no) (category/int64).
        'atraso_15',
        # period of the day (morning, afternoon, night) in which the flight was scheduled to depart. (category)
        'periodo_dia',
        # high season (1 = yes, 0 = no) in which the flight was scheduled to depart. (category/int64)
        'temporada_alta',
    ]]


# ---------------------------------------------------------------------------------------------------------------


def make_pretty(
    df: pd.DataFrame,
    caption: Union[str, Tuple[str, str]] = '',
    formatter: Optional[Union[Dict, Callable]] = None,
) -> Styler:
    """Set the table styles and caption of the given Styler object.

    Args:
        df (pd.DataFrame): DataFrame to be styled. Its styler attribute will be modified in-place and returned.
        caption (Union[str, Tuple[str, str]]): Caption of the table. Defaults to ''.
        formatter (Optional[Union[Dict, Callable]]): Formatter to be used for the table. Defaults to None.

    Returns:
        Styler: The styler object of the given DataFrame with the table styles and caption set as specified.
    """
    return df.style.set_table_styles(TABLE_STYLES).set_caption(caption).format(formatter)


# ---------------------------------------------------------------------------------------------------------------


def summary(data: pd.DataFrame, name: str = 'data') -> None:
    """Display a summary of the data.

    Args:
        data (pd.DataFrame): Data to be summarized.
        name (str, optional): Name of the data. Defaults to 'data'.
    """
    display(Markdown(f'## Summary of the {name}'))

    # head
    display(
        # data
        data
        # first 5 rows
        .head()
        # style the dataframe
        .style
        # set table styles
        .set_table_styles(TABLE_STYLES)
        # set caption
        .set_caption(f'First 5 rows of the {name}')
        # display the table
    )

    # tail
    display(
        # data
        data
        # last 5 rows
        .tail()
        # style the dataframe
        .style
        # set table styles
        .set_table_styles(TABLE_STYLES)
        # set caption
        .set_caption(f'Last 5 rows of the {name}')
        # display the table
    )

    # info
    display(Markdown(f'### Information about the {name}'))
    data.info()

    # summarize
    summarize(data=data, name=name)

    return None


def summarize(data: pd.DataFrame, name: str = 'data') -> None:
    """Display a summary of the data (categorical, numerical and datetime columns).

    Args:
        data (pd.DataFrame): Data to be summarized.
        name (str, optional): Name of the data. Defaults to 'data'.
    """
    if not data.select_dtypes(include=['category']).empty:
        # if there are categorical features
        summarize_categorical(data=data, name=name)

    if not data.select_dtypes(include=['number']).empty:
        # if there are numerical features
        summarize_numerical(data=data, name=name)

    if not data.select_dtypes(include=['datetime']).empty:
        # if there are datetime features
        summarize_datetime(data=data, name=name)

    return None


def summarize_categorical(data: pd.DataFrame, name: str = 'data') -> None:
    """Display a summary of the data (categorical columns).

    Args:
        data (pd.DataFrame): Data to be summarized.
        name (str, optional): Name of the data. Defaults to 'data'.
    """
    display(
        # make a copy of the dataframe to avoid modifying the original one
        data.copy(deep=True)
        # select categorical features
        .select_dtypes(include=['category'])
        # compute descriptive statistics
        .describe()
        # transpose the table to make it more readable
        .T
        # compute the relative frequency of the most frequent category
        .assign(most_freq_rel_freq=lambda df: df['freq'] / df['count'])
        # sort the table by the relative frequency of the most frequent category, in descending order
        .sort_values(by='most_freq_rel_freq', ascending=False)
        # style the dataframe
        .style
        # set table styles
        .set_table_styles(TABLE_STYLES)
        # set caption
        .set_caption(f'Descriptive statistics of categorical features of the {name}')
        # format the relative frequency of the most frequent category as a percentage
        .format(formatter={'most_freq_rel_freq': '{:.3%}'})
        # highlight quantiles
        .pipe(
            lambda styler, subset, colors, axis:
            # highlight quartiles
            styler
            #   1st quartile (green)
            .highlight_quantile(subset=subset, color=colors[0], q_left=0.00, q_right=0.25, axis=axis)
            #   2nd quartile (yellow)
            .highlight_quantile(subset=subset, color=colors[1], q_left=0.25, q_right=0.50, axis=axis)
            #   3rd quartile (orange)
            .highlight_quantile(subset=subset, color=colors[2], q_left=0.50, q_right=0.75, axis=axis)
            #   4th quartile (red)
            .highlight_quantile(subset=subset, color=colors[3], q_left=0.75, q_right=1.00, axis=axis),
            # subset of columns to highlight
            subset=[
                'unique'
                # 'most_freq_rel_freq',
            ],
            # colors to use for highlighting quartiles (1st: green, 2nd: yellow, 3rd: orange, 4th: red)
            colors=['lightgreen', 'lightyellow', 'rgb(255, 230, 153)', 'lightcoral'],
            # axis along which to compute the quantiles (0: index, 1: columns)
            axis=0,
        )
        # add an in-cell barplot to visualize the relative frequency of the most frequent category
        .bar(
            subset=['most_freq_rel_freq'],
            # color='#5fba7d',  # green
            # color='#d65f5f',  # red
            color='#2b8cbe',  # blue
            vmin=0,
            vmax=1,
        )
        # display the table
    )

    return None


def summarize_numerical(data: pd.DataFrame, name: str = 'data') -> None:
    """Display a summary of the data (numerical columns).

    Args:
        data (pd.DataFrame): Data to be summarized.
        name (str, optional): Name of the data. Defaults to 'data'.
    """
    display(
        # make a copy of the dataframe to avoid modifying the original one
        data.copy(deep=True)
        # select numerical features
        .select_dtypes(include=['number'])
        # compute descriptive statistics
        .describe()
        # transpose the table to make it more readable
        .T
        # style the dataframe
        .style
        # set table styles
        .set_table_styles(TABLE_STYLES)
        # set caption
        .set_caption(f'Descriptive statistics of numerical features of the {name}')
        # display the table
    )

    return None


def summarize_datetime(data: pd.DataFrame, name: str = 'data') -> None:
    """Display a summary of the data (datetime columns).

    Args:
        data (pd.DataFrame): Data to be summarized.
        name (str, optional): Name of the data. Defaults to 'data'.
    """
    display(
        # make a copy of the dataframe to avoid modifying the original one
        data.copy(deep=True)
        # select datetime features
        .select_dtypes(include=['datetime'])
        # compute descriptive statistics
        .describe(
            # if datetime_is_numeric is True, then 'first', 'last' columns will not be present in the output dataframe
            # datetime_is_numeric=True,
            #
        )
        # transpose the table to make it more readable
        .T
        # style the dataframe
        .style
        # set table styles
        .set_table_styles(TABLE_STYLES)
        # set caption
        .set_caption(f'Descriptive statistics of datetime features of the {name}')
        # highlight the minimum and maximum values
        .highlight_min(
            subset=[
                'first',
                'last',
            ],
            # subset=['min', 'max',],
            color='#d65f5f',  # red
        ).highlight_max(
            subset=[
                'first',
                'last',
            ],
            # subset=['min', 'max',],
            color='#5fba7d',  # green
        )
        # display the table
    )

    return None


# ---------------------------------------------------------------------------------------------------------------


def scipy_describe(
    data: pd.DataFrame,
    *,
    ddof=1,
    bias=True,
    nan_policy='propagate',
) -> pd.DataFrame:
    """Compute several descriptive statistics of a DataFrame. This function wraps the `scipy.stats.describe` function.
    It computes the 'min', 'max', 'mean', 'variance', 'skewness', 'kurtosis' statistics.

    Args:
        data: The data to compute the statistics of.

        ddof: The delta degrees of freedom.

        bias: Whether the skewness and kurtosis calculations are corrected for statistical bias or not.

        nan_policy: The policy to use when encountering NaN values.
    """
    axis = 0  # 0: index (rows), 1: columns, None: both

    # named tuple containing the statistics
    # DescribeResult = namedtuple('DescribeResult', ('nobs', 'minmax', 'mean', 'variance', 'skewness', 'kurtosis'))
    describe_result = stats.describe(
        a=data.copy(deep=True).to_numpy(),
        # kwargs
        axis=axis,
        ddof=ddof,
        bias=bias,
        nan_policy=nan_policy,
    )

    return (
        # load the statistics into a dataframe
        pd.DataFrame(
            # get the columns names from the dataframe
            columns=data.columns.to_list(),
            data=[
                # describe_result.nobs,  # number of observations (number of rows)
                describe_result.minmax[0],  # min
                describe_result.minmax[1],  # max
                describe_result.mean,
                describe_result.variance,
                describe_result.skewness,
                describe_result.kurtosis,
            ],
            index=[
                # 'nobs', # 'number of observations' (number of rows)
                'min',
                'max',
                'mean',
                'variance',
                'skewness',
                'kurtosis',
            ],
        )
        # transpose the dataframe for better readability
        .T
        # return the dataframe
    )
