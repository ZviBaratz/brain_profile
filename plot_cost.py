import numpy as np
import pandas as pd
import pickle

from bokeh.io import curdoc
from bokeh.layouts import row, widgetbox
from bokeh.models import ColumnDataSource, CDSView, BooleanFilter, FactorRange
from bokeh.models.widgets import MultiSelect
from bokeh.plotting import figure


RESULTS_PATH = "/home/flavus/Projects/reid/results.pkl"


def read_results(path: str):
    try:
        with open(path, "rb") as results_content:
            return pickle.load(results_content)
    except Exception:
        raise FileNotFoundError("Failed to load results!")


df = read_results(RESULTS_PATH)


def query_results(
    results: pd.DataFrame,
    subject_id: str = None,
    cost_function: str = None,
    metric: str = "Mutual Information",
) -> pd.DataFrame:
    subject_id = subject_id or slice(None)
    cost_function = cost_function or slice(None)
    metric = metric or slice(None)
    return results.sort_index().loc[(subject_id, cost_function, metric), :]


def create_column_data_source(dataframe: pd.DataFrame) -> ColumnDataSource:
    subject_id = list(dataframe.index.get_level_values(0))
    cost_function = list(dataframe.index.get_level_values(1))
    metric = list(dataframe.index.get_level_values(2))
    value = [value[0] for value in dataframe.values]
    return ColumnDataSource(
        data=dict(
            subject_id=subject_id,
            cost_function=cost_function,
            metric=metric,
            value=value,
        )
    )


def extract_cost_function_results(dataframe: pd.DataFrame, cost_function: str) -> list:
    results = query_results(dataframe, cost_function=cost_function)
    return [value[0] for value in results.values]


def make_plot(title, hist, edges, x):
    p = figure(title=title, tools="", background_fill_color="#fafafa")
    p.quad(
        top=hist,
        bottom=0,
        left=edges[:-1],
        right=edges[1:],
        fill_color="navy",
        line_color="white",
        alpha=0.5,
    )
    p.y_range.start = 0
    p.legend.location = "center_right"
    p.legend.background_fill_color = "#fefefe"
    p.xaxis.axis_label = "x"
    p.yaxis.axis_label = "Pr(x)"
    p.grid.grid_line_color = "white"
    return p


values = extract_cost_function_results(df, "Correlation Ratio")
hist, edges = np.histogram(values, density=True, bins=50)
x = np.linspace(0, 1, 1000)

plot = make_plot("Correlation Ratio", hist, edges, x)

layout = row(plot)
curdoc().add_root(layout)
