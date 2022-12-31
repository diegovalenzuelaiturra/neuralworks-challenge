"""Charts module."""
from contextlib import contextmanager
import logging
from typing import Optional, TYPE_CHECKING

import pandas as pd
import plotly.express as px

from neuralworks.constants import get_custom_plotly_figure_size

if TYPE_CHECKING:
    from typing import Literal

logger: logging.Logger = logging.getLogger(__name__)

ALLOWED_PANDAS_PLOTTING_BACKENDS = [
    'matplotlib',
    'plotly',
]

CUSTOM_PLOTLY_FIGURE_SIZE = get_custom_plotly_figure_size()


# 'pandas-profiling' is not compatible with the pandas plotting backend 'plotly'
# So, we need to set the backend to 'matplotlib' while using 'pandas-profiling'
# And then set it back to its original value when we are done
@contextmanager
def pandas_plotting_backend(backend: "Literal['matplotlib', 'plotly']"):
    """Context manager to temporarily set the pandas plotting backend.

    Args:
        backend (str): The pandas plotting backend to use. Currently, only 'matplotlib' and 'plotly' are supported.

    Raises:
        ValueError: If the backend is not supported.

        Exception: If an error occurs while setting the backend.
    """
    if backend not in ALLOWED_PANDAS_PLOTTING_BACKENDS:
        logger.error('Invalid backend: %s. Must be one of %s.', backend, ALLOWED_PANDAS_PLOTTING_BACKENDS)
        raise ValueError(f'Invalid backend: {backend}. Must be one of {ALLOWED_PANDAS_PLOTTING_BACKENDS}.')

    # get the current pandas plotting backend
    original_backend = pd.options.plotting.backend

    # set the pandas plotting backend to the specified backend
    pd.options.plotting.backend = backend

    try:
        # yield control back to the context
        yield
    except Exception as e:
        logger.error('Error while using pandas plotting backend %s: %s', backend, e)
        raise e
    finally:
        # set the pandas plotting backend back to the original backend before exiting the context
        pd.options.plotting.backend = original_backend


def compute_delay_ratio_per_column(
    df: pd.DataFrame,
    column: str,
    column_descriptive_name: Optional[str] = None,
    column_values_mapping: Optional[dict] = None,
    *,
    sorted: bool = True,
    ascending: bool = True,
    reset_index: bool = True,
    rename_columns: bool = True,
    set_index: bool = True,
    inplace: bool = False,
    dropna: bool = True,
    drop_duplicates: bool = True,
):
    """Compute the delay ratio per 'column'.

    Args:
        df (pd.DataFrame): dataframe

        column (str): column name to group by.

        column_descriptive_name (str, optional): rename the column to have a more descriptive name. Defaults to None.

        column_values_mapping (dict, optional): map the column values to be more descriptive. Defaults to None.

    Returns:
        pd.DataFrame: dataframe with the delay ratio per 'column'.
    """
    # sorted (bool, optional): sort by the ratio of delayed flights, in descending order. Defaults to True.
    # ascending (bool, optional): sort by the ratio of delayed flights, in ascending order. Defaults to True.
    # reset_index (bool, optional): reset the index to have the 'column' as a column. Defaults to True.
    # rename_columns (bool, optional): rename the columns to have a more meaningful name for the report and the plots (sum -> #delayed, count -> #total, mean -> ratio). Defaults to True.
    # set_index (bool, optional): set the index to have the 'column' as the index. Defaults to True.
    # inplace (bool, optional): modify the original dataframe. Defaults to False.
    # dropna (bool, optional): drop rows with NaN values. Defaults to True.
    # drop_duplicates (bool, optional): drop duplicate rows. Defaults to True.

    if not inplace:
        # create a copy of the dataframe to avoid modifying the original dataframe
        df = df.copy(deep=True)

    # transform the 'atraso_15' column to int (instead of category) to be able to perform the aggregation
    df['atraso_15'] = df['atraso_15'].astype('int')

    # calculate the delay count, sum and mean (ratio of delayed flights) per 'column'
    df = df.groupby(column)['atraso_15'].agg(['sum', 'count', 'mean'])

    # sort by the ratio of delayed flights, in descending order
    if sorted:
        df = df.sort_values(by='mean', ascending=ascending)

    if reset_index:
        # reset the index to have the 'column' as a column
        df = df.reset_index()

    if rename_columns:  # TODO: all if/else conditions are not considered yet
        # rename the columns to have a more meaningful name for the report and the plots
        # (sum -> #delayed, count -> #total, mean -> ratio)
        df = df.rename(columns={'sum': '#delayed', 'count': '#total', 'mean': 'ratio'})

    if column_descriptive_name:
        # rename the 'column' to have a more meaningful name for the report and the plots
        # (column -> column_descriptive_name)
        df = df.rename(columns={
            column: column_descriptive_name,
        })
        # update the 'column' variable to use the new column name
        column = column_descriptive_name

    if set_index:
        # set the 'column' as the index
        df = df.set_index(column)

    if column_values_mapping:
        # map the 'column' values to be more descriptive
        df = df.rename(index=column_values_mapping)

    if dropna:
        # drop rows with NaN values
        df = df.dropna()

    if drop_duplicates:
        # drop duplicate rows
        df = df.drop_duplicates()

    if column_descriptive_name:
        # rename the index to have a more meaningful name for the report and the plots
        # (column -> column_descriptive_name)
        df = df.rename_axis(column_descriptive_name)

    return df


def plot_delay_ratio_per_column(
    df: pd.DataFrame,
    column_descriptive_name: Optional[str] = None,
    *,
    delay_ratio_column: str = 'ratio',  # tasa atraso
    total_flights_column: str = '#total',  # total vuelos
    delayed_flights_column: str = '#delayed',  # vuelos atrasados
    height_multiplier: float = 1.0,
):
    """Plot the delay ratio per 'column'.

    Args:
        df (pd.DataFrame): dataframe

        column_descriptive_name (str, optional): rename the column to have a more descriptive name. Defaults to None.

        delay_ratio_column (str, optional): column name for the delay ratio. Defaults to 'ratio'.
        total_flights_column (str, optional): column name for the number of flights. Defaults to '#total'.
        delayed_flights_column (str, optional): column name for the number of delayed flights. Defaults to '#delayed'.

        height_multiplier (float, optional): multiply the height of the plot by this factor. Defaults to 1.0.
    """
    # plot the ratio of delayed flights per 'column' (sorted by the ratio of delayed flights)
    fig = px.bar(
        df,
        x=delay_ratio_column,
        y=df.index,
        orientation='h',
        title=f'Tasa de vuelos retrasados por {column_descriptive_name}',
        labels={
            delay_ratio_column: 'Tasa de vuelos retrasados',
            delayed_flights_column: 'Número de vuelos retrasados',
            total_flights_column: 'Número de vuelos totales',
            'index': column_descriptive_name,
        },
        width=CUSTOM_PLOTLY_FIGURE_SIZE[0],
        height=CUSTOM_PLOTLY_FIGURE_SIZE[1] * height_multiplier,
        range_x=[0, 1],
        color=delayed_flights_column,
        text=delayed_flights_column,
        hover_data={
            delay_ratio_column: ':.3f',  # ratio of delayed flights per 'column' (using 3 decimal places)
            total_flights_column: True,  # number of total flights per 'column'
            delayed_flights_column: True,  # number of delayed flights per 'column'
        },
    )
    fig.show()


def plot_delay_ratios(df: pd.DataFrame):
    """Plot the delay ratios per column."""
    df = df.copy(deep=True)

    # Destination
    plot_delay_ratio_per_column(
        df=compute_delay_ratio_per_column(
            df=df,
            column='SIGLADES',  # 'SIGLADES' is the column name for the destination
            column_descriptive_name='Destino',
            column_values_mapping=None,
        ),
        column_descriptive_name='Destino',
        height_multiplier=1.8,
    )

    # Airline
    plot_delay_ratio_per_column(
        df=compute_delay_ratio_per_column(
            df=df,
            column='Emp-O',  # 'Emp-O' is the column name for the airline
            column_descriptive_name='Aerolínea',
            column_values_mapping=None,
        ),
        column_descriptive_name='Aerolínea',
        height_multiplier=1.0,
    )

    # Month
    plot_delay_ratio_per_column(
        df=compute_delay_ratio_per_column(
            df=df,
            column='MES',  # 'MES' is the column name for the month
            column_descriptive_name='Mes',
            column_values_mapping={
                '1': 'Enero',
                '2': 'Febrero',
                '3': 'Marzo',
                '4': 'Abril',
                '5': 'Mayo',
                '6': 'Junio',
                '7': 'Julio',
                '8': 'Agosto',
                '9': 'Septiembre',
                '10': 'Octubre',
                '11': 'Noviembre',
                '12': 'Diciembre',
            },
        ),
        column_descriptive_name='Mes',
        height_multiplier=0.7,
    )

    # Day of week
    plot_delay_ratio_per_column(
        df=compute_delay_ratio_per_column(
            df=df,
            column='DIANOM',  # 'DIANOM' is the column name for the day of week
            column_descriptive_name='Día de la semana',
            column_values_mapping=None,
        ),
        column_descriptive_name='Día de la semana',
        height_multiplier=0.6,
    )

    # Season
    plot_delay_ratio_per_column(
        df=compute_delay_ratio_per_column(
            df=df,
            column='temporada_alta',  # 'temporada_alta' is the column name for the season
            column_descriptive_name='Temporada',
            column_values_mapping={
                0: "Baja",
                1: "Alta",
            },
        ),
        column_descriptive_name='Temporada',
        height_multiplier=0.5,
    )

    # Flight type
    plot_delay_ratio_per_column(
        df=compute_delay_ratio_per_column(
            df=df,
            column='TIPOVUELO',  # 'TIPOVUELO' is the column name for the flight type
            column_descriptive_name='Tipo de vuelo',
            column_values_mapping={
                'N': "Nacional",
                'I': "Internacional",
            },
        ),
        column_descriptive_name='Tipo de vuelo',
        height_multiplier=0.5,
    )
