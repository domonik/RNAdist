from RNAdist import _version
import plotly.io as pio
import plotly.graph_objs as go
import numpy as np
import dash_bootstrap_components as dbc
import dash
from dash import dcc
from dash import html
from dash import callback_context
from dash.dependencies import Input, Output, State, ALL
import os
import pickle
import base64
from dash.exceptions import PreventUpdate

__version__ = _version.get_versions()["version"]
FILEDIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(FILEDIR, "assets")

LOGO = os.path.join(ASSETS_DIR, "RNAdistVisualizer_dark.svg")
assert os.path.exists(LOGO)
encoded_img = base64.b64encode(open(LOGO, 'rb').read())


app = dash.Dash(
    "RNAdist Dashboard",
    external_stylesheets=[dbc.themes.DARKLY],
    assets_url_path=ASSETS_DIR,
    assets_folder=ASSETS_DIR,
    index_string=open(os.path.join(ASSETS_DIR, "index.html")).read(),
)

pio.templates["plotly_white"].update(
    {
        "layout": {
            # e.g. you want to change the background to transparent
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": " rgba(0,0,0,0)",
            "font": dict(color="white"),
        }
    }
)


def _header_layout():
    svg = 'data:image/svg+xml;base64,{}'.format(encoded_img.decode())
    header = html.Div(
        html.Div(
            html.Img(src=svg, style={"width": "30%", "min-width": "300px"}, className="p-3" ),
            className="databox",
            style={"text-align": "center"},
        ),
        className="col-12 p-1 justify-content-center"
    )
    return header


def _scatter_from_data(data, key, line: int = 0, start: int = 0):
    data_row = data[key][line, start:]
    x = np.arange(start, len(data_row)+start)
    scatter = go.Scatter(
        y=data_row,
        x=x,
        line={"width": 4, "color": "#ff8add"}

    )
    fig = go.Figure()
    fig.layout.template = "plotly_white"
    fig.add_trace(scatter)
    fig.add_vrect(
        x0=max(line-.5, 0), x1=min(line+.5, len(data_row)+start - 1),
        fillcolor="#AAE4EE", opacity=0.5,
        layer="below", line_width=0,
    ),
    return fig


def _heatmap_from_data(data, key):
    fig = go.Figure()
    fig.layout.template = "plotly_white"
    heatmap = go.Heatmap(
        z=data[key]
    )
    fig.add_trace(heatmap)
    fig['layout']['yaxis']['autorange'] = "reversed"
    return fig


def _range_button(range_max):
    data = [
            html.Div(
                [
                    dbc.Input(type="number", min=0, max=range_max, step=1, id="input-line"),
                ],
                className="justify-content-center",
            ),
    ]
    return data


def _distance_box():

    d_box = html.Div(
        [
            html.Div(
                [
                    dcc.Graph(id="distance-graph", style={"height": "385px"})
                ],
                className="databox",
            )
        ],
        className="col-12 p-1 justify-content-center"
    )
    return d_box


def _heatmap_box():
    d_box = html.Div(
        [
            html.Div(
                [
                    dcc.Graph(id="heatmap-graph")
                ],
                className="databox"
            )
        ],
        className="col-6 p-1 justify-content-center"
    )
    return d_box


def _selector(data):
    data = list(data)
    d_box = html.Div(
        [
            html.Div(
                [
                    html.Div(
                        html.Div(
                            html.H4("Selector", style={"text-align": "center"}),
                            className="col-12 justify-content-center"
                        ),
                        className="row justify-content-center p-4"
                    ),
                    html.Div(
                        [
                            html.Div(
                                html.Span("Sequence", style={"text-align": "center"}),
                                className="col-3 justify-content-center align-self-center"
                            ),
                            html.Div(
                                dcc.Dropdown(
                                    data[0:100], data[0],
                                    className="justify-content-center",
                                    id="sequence-selector"
                                ),
                                className="col-7 justify-content-center",
                            ),
                        ],
                        className="row justify-content-center p-2"
                    ),
                    html.Div(
                        [
                            html.Div(
                                html.Span("Nucleotide index", style={"text-align": "center"}),
                                className="col-3 justify-content-center align-self-center"
                            ),
                            html.Div(
                                _range_button(10),
                                className="col-7 justify-content-center",
                                id="range-buttons"
                            ),
                        ],
                        className="row justify-content-center p-2"
                    ),
                ],
                className="databox justify-content-center"
            )
        ],
        className="col-6 p-1 justify-content-center"
    )
    return d_box


def get_app_layout(dash_app: dash.Dash):
    dash_app.layout = html.Div(
        [
            dcc.Location(id="url", refresh=False),
            html.Div(
                _header_layout(),
                className="row justify-content-center"
            ),
            html.Div(
                _distance_box(),
                className="row justify-content-center"
            ),
            html.Div(
                [
                    _heatmap_box(),
                    _selector(data),
                ],
                className="row justify-content-center"
            )

        ]
    )


@app.callback(
    Output("sequence-selector", "options"),
    Input("sequence-selector", "search_value")
)
def update_options(search_value):
    if not search_value:
        raise PreventUpdate
    options = [o for o in data if search_value in o][0:100]
    return options


@app.callback(
        Output("heatmap-graph", "figure"),
        Input("sequence-selector", "value"),
)
def _update_heatmap(value):
    fig = _heatmap_from_data(data, value)
    return fig


@app.callback(
    Output("input-line", "value"),
    [Input("heatmap-graph", "clickData"),
     Input("sequence-selector", "value")]

)
def _update_on_click(value, key):
    if value is not None:
        line = value["points"][0]["y"]
    else:
        line = 0
    return line

@app.callback(
    Output("range-buttons", "children"),
    Input("sequence-selector", "value")

)
def _update_range_buttons(key):
    cur_data = data[key]
    range_max = len(cur_data)
    return _range_button(range_max)



@app.callback(
    Output("distance-graph", "figure"),
    [
        Input("input-line", "value"),
        Input("sequence-selector", "value")]

)
def _update_plot(line, key):
    if line is None:
        line = 0
    fig = _scatter_from_data(data, key, line, 0)
    return fig


def run_visualization(args):
    global data
    print("loading data")
    with open(args.input, "rb") as handle:
        data = pickle.load(handle)
    print("finished loading going to start server")
    get_app_layout(app)
    app.run(debug=False, port=args.port, host=args.host)

data = {}


