from RNAdist import _version
import plotly.io as pio
import plotly.graph_objs as go
import numpy as np
import dash_bootstrap_components as dbc
import dash
from dash import dcc
from dash import html, callback_context
from dash.dependencies import Input, Output, State
import os
import pickle
import base64
from dash.exceptions import PreventUpdate
import pandas as pd
from Bio import SeqIO
import plotly

__version__ = _version.get_versions()["version"]
FILEDIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(FILEDIR, "assets")

LOGO = os.path.join(ASSETS_DIR, "RNAdistVisualizer_dark.svg")
assert os.path.exists(LOGO)
encoded_img = base64.b64encode(open(LOGO, 'rb').read())

COLORS = ["#ff8add"] + list(plotly.colors.qualitative.Light24)

app = dash.Dash(
    "RNAdist Dashboard",
    title="RNAdist Visualizer",
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


def _scatter_from_data(cur_data, key, line: int = 0, start: int = 0):
    fig = go.Figure()
    fig.layout.template = "plotly_white"
    cur_data = cur_data[key]
    for cidx, (ip_name, value) in enumerate(cur_data.items()):
        data_row = value[line, start:]
        x = np.arange(start, len(data_row)+start)
        if fasta:
            customdata = list(fasta[key][start:len(data_row)+start])
            hovertemplate = f'i:{line} [{fasta[key][line]}]<br>j:%{{x}} [%{{customdata}}]<br>distance: %{{y}}'
        else:
            customdata = None
            hovertemplate = f'i:{line} <br>j:%{{x}}<br>distance: %{{y}}'
        scatter = go.Scatter(
            y=data_row,
            x=x,
            line={"width": 4, "color": COLORS[cidx]},
            customdata=customdata,
            hovertemplate=hovertemplate,
            name=ip_name

        )

        fig.add_trace(scatter)
    fig.add_vrect(
        x0=max(line-.5, 0), x1=min(line+.5, len(data_row)+start - 1),
        fillcolor="#AAE4EE", opacity=0.5,
        layer="below", line_width=0,
    ),
    key = key if len(key) <= 30 else key[0:30] + "..."
    fig.update_layout(
        title={"text": f"Expected Distance<br>{key}", "font": {"size": 20}},
        xaxis_title="Nucleotide j",
        yaxis_title="Expected Distance",
        title_x=0.5
    )
    return fig


def _heatmap_from_data(data, key, file):
    fig = go.Figure()
    fig.layout.template = "plotly_white"
    d = data[key][file]
    if fasta:
        a = np.asarray(list(fasta[key]))[:len(d)]
        customdata = np.asarray(np.meshgrid(a, a)).T.reshape(-1, 2).reshape(len(a), len(a), -1)
        hovertemplate = 'i:%{y} [%{customdata[0]}] <br><b>j:%{x} [%{customdata[1]}] </b><br>distance: %{z}'
    else:
        hovertemplate = 'i:%{y} <br><b>j:%{x}</b><br>distance: %{z}'
        customdata = None
    heatmap = go.Heatmap(
        z=d,
        hovertemplate=hovertemplate,
        customdata=customdata,
        name=""
    )
    fig.add_trace(heatmap)
    fig['layout']['yaxis']['autorange'] = "reversed"
    fig.update_layout(
        title={"text": "Distance Heatmap", "font": {"size": 20}},
        xaxis_title="Nucleotide j",
        yaxis_title="Nucleotide i",
        title_x=0.5,
        xaxis_side="top",
    )
    fig.update_xaxes(
        title_standoff=0
    )
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
                    dcc.Graph(id="distance-graph", style={"height": "370px"})
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
        className="col-12 col-md-6 p-1 justify-content-center order-md-first"
    )
    return d_box


def _selector(data):
    sel_data = list(data)
    data_files = list(data[sel_data[0]])
    d_box = html.Div(
        [
            html.Div(
                [
                    html.Div(
                        html.Div(
                            html.H4("Selector", style={"text-align": "center"}),
                            className="col-12 justify-content-center"
                        ),
                        className="row justify-content-center p-2 p-md-2"
                    ),
                    html.Div(
                        [
                            html.Div(
                                html.Span("Sequence", style={"text-align": "center"}),
                                className="col-10 col-md-3 justify-content-center align-self-center"
                            ),
                            html.Div(
                                dcc.Dropdown(
                                    sel_data[0:100], sel_data[0],
                                    className="justify-content-center",
                                    id="sequence-selector"
                                ),
                                className="col-10 col-md-7 justify-content-center",
                            ),
                        ],
                        className="row justify-content-center p-2"
                    ),
                    html.Div(
                        [
                            html.Div(
                                html.Span("Heatmap Prediction", style={"text-align": "center"}),
                                className="col-10 col-md-3 justify-content-center align-self-center"
                            ),
                            html.Div(
                                dcc.Dropdown(
                                    data_files[0:100], data_files[0],
                                    className="justify-content-center",
                                    id="data-selector"
                                ),
                                className="col-10 col-md-7 justify-content-center",
                            ),
                        ],
                        className="row justify-content-center p-2"
                    ),
                    html.Div(
                        [
                            html.Div(
                                html.Span("Nucleotide index", style={"text-align": "center"}),
                                className="col-10 col-md-3 justify-content-center align-self-center"
                            ),
                            html.Div(
                                _range_button(10),
                                className="col-10 col-md-7 justify-content-center",
                                id="range-buttons"
                            ),
                        ],
                        className="row justify-content-center p-2"
                    ),
                    html.Div(
                        [
                            html.Div(
                                dbc.Button("Export Heatmap to TSV", style={"text-align": "center", "width": "100%"}, id="open-modal"),
                                className="col-10 justify-content-center align-self-center"
                            ),
                            _modal_tsv_download()

                        ],
                        className="row justify-content-center pb-4 p-2 pb-md-2"
                    ),
                    html.Div(
                        [
                            # This is the Div where the sequence will be placed
                        ],
                        className="row justify-content-center pb-4 p-2 pb-md-2",
                        id="fasta-seq"
                    ),
                ],
                className="databox justify-content-center"
            )
        ],
        className="col-12 col-md-6 p-1 justify-content-center"
    )
    return d_box


def _modal_tsv_download():
    modal = dbc.Modal(
        [
            dbc.ModalHeader("Select file Name"),
            dbc.ModalBody(
                [
                    html.Div(
                        [
                            html.Div(dbc.Input("named-download",),
                                        className=" col-9"),
                            dbc.Button("Download", id="download-tsv-button", className="btn btn-primary col-3"),
                            dcc.Download(id="download-tsv")
                        ],
                        className="row justify-content-around",
                    )
                ]
            ),
            dbc.ModalFooter(
                dbc.Button("Close", id="close", className="ml-auto",
                           n_clicks=0)),
        ],
        id="modal",
    )
    return modal


def _get_app_layout(dash_app: dash.Dash):
    dash_app.layout = html.Div(
        [
            dcc.Location(id="url", refresh=False),
            html.Div(
                _header_layout(),
                className="row m-1 justify-content-center align-items-center"
            ),
            html.Div(
                _distance_box(),
                className="row m-1 justify-content-center"
            ),
            html.Div(
                [
                    _selector(data),
                    _heatmap_box(),

                ],
                className="row m-1  justify-content-center"
            )

        ]
    )


@app.callback(
    Output("sequence-selector", "options"),
    Input("sequence-selector", "search_value")
)
def _update_options(search_value):
    if not search_value:
        raise PreventUpdate
    options = [o for o in data if search_value in o][0:100]
    return options


@app.callback(
        Output("heatmap-graph", "figure"),
        [
            Input("sequence-selector", "value"),
            Input("data-selector", "value")
        ]
)
def _update_heatmap(key, file):
    fig = _heatmap_from_data(data, key, file)
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
    cur_data = data[key][list(data[key].keys())[0]]
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


@app.callback(
    [
        Output("modal", "is_open"),
        Output("named-download", "value")
     ],
    [
        Input("open-modal", "n_clicks"),
        Input("close", "n_clicks"),
        Input("download-tsv-button", "n_clicks"),
    ],
    [State("modal", "is_open"),
     State("sequence-selector", "value")
     ],
    prevent_initial_call=True

)
def _toggle_modal(n1, n2, n3, is_open, seqname):
    filename = seqname + ".tsv"
    if n1 or n2 or n3:
        return not is_open, filename
    return is_open, filename


@app.callback(
    Output("download-tsv", "data"),
    [
        Input("download-tsv-button", "n_clicks"),

    ],
    [
        State("named-download", "value"),
        State("sequence-selector", "value")
    ],
    prevent_initial_call=True
)
def _download_tsv(n_clicks, filename, key):
    filename = os.path.basename(filename)
    to_download = data[key]
    df = pd.DataFrame(to_download)
    return dcc.send_data_frame(df.to_csv, filename, sep="\t")


@app.callback(
    Output("fasta-seq", "children"),
    Input("sequence-selector", "value")
)
def _add_fasta_sequence(key):
    if fasta:
        seq = fasta[key]
        md = f"\>{key}\n\n{seq}"
        div = html.Div(
            dcc.Markdown(md),
            style={"maxHeight": "160px"},
            className="col-10 justify-content-center align-self-center fasta-panel"
        )

    else:
        div = html.Div()
    return div





def run_visualization(args):
    """Wrapper for running visualization Dashboard

    Args:
        args (argparse.Namespace): cli args containing input, port and host
    """
    global data
    print("loading data")
    for ip in args.input:
        with open(ip, "rb") as handle:
            d = pickle.load(handle)
            for k, v in d.items():
                if k not in data:
                    data[k] = {}
                data[k][ip] = v

    if args.fasta:
        seqs = {sr.description: str(sr.seq) for sr in SeqIO.parse(args.fasta, "fasta")}
        global fasta
        fasta = seqs
    print("finished loading going to start server")
    _get_app_layout(app)
    app.run(debug=False, port=args.port, host=args.host)

data = {}
fasta = {}

if __name__ == '__main__':
    import argparse
    args = argparse.Namespace()
    args.input = ["foofile", "foofile2"]
    args.port = "8080"
    args.host = "0.0.0.0"
    args.fasta = "foofasta.fa"
    run_visualization(args)

