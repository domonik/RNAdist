import RNA
import dash
from dash import callback, html, clientside_callback, Input, Output, dcc, dash_table, State, Patch, clientside_callback
import dash_bootstrap_components as dbc
from RNAdist.dashboard import DATABASE_FILE, CACHE, LAYOUT, DARK_LAYOUT, COLORS
from RNAdist.sampling.ed_sampling import distance_histogram
from RNAdist.sampling.ed_sampling import bytes_to_structure
from RNAdist.plots.sampling_plots import distance_histo_from_matrix, expected_median_distance_maxtrix, empty_figure, plot_distances_with_running_j
import sqlite3
import numpy as np
import io
from RNAdist.dashboard.helpers import get_structures_and_length_for_hash, matrix_from_hash, get_structures_by_ids
import zlib
import uuid
import dash_bio as dashbio
from dash import ClientsideFunction
import zlib


dash.register_page(__name__, path='/visualization', name="Visualization")


def get_selected_sequence():
    sel_sequence = dbc.Col(
        dbc.Card(
            [
                dbc.CardHeader(
                    dbc.Row(
                        [
                            dbc.Col(html.H5("Sequence"), width=3),
                            dbc.Col(dcc.Dropdown(
                                style={"width": "100%"},
                                id="seqid",

                            ), width=3, className="d-flex align-items-center"),

                        ]

                    ),

                ),

            ],

        )
    )
    return sel_sequence

def get_distance_distrubution_box():
    distance_distribution_box = dbc.Col(
        dbc.Card(
            [
                dbc.CardHeader(
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.H5("Distance Distribution"),
                                    html.Span(
                                        [
                                            html.I(className="fas fa-xl fa-question-circle fa px-2",
                                                   id="filter-tip"),
                                            dbc.Tooltip(
                                                "Selecting only i will show a summary for a running j. Selecting also j "
                                                "results in a histogram of distances",
                                                target="filter-tip"
                                            ),
                                        ],
                                    )
                                ], width=4,
                                className="d-flex align-items-center"
                            ),
                            dbc.Col(html.Span("i"), width=1,
                                    className="d-flex align-items-center justify-content-end"),
                            dbc.Col(dcc.Dropdown(
                                style={"width": "100%"},
                                id="nt-i",

                            ), width=2, className="d-flex align-items-center"),
                            dbc.Col(html.Span("j"), width=1,
                                    className="d-flex align-items-center justify-content-end"),
                            dbc.Col(dcc.Dropdown(
                                style={"width": "100%"},
                                id="nt-j",

                            ), width=2, className="d-flex align-items-center"),

                        ]

                    ),

                ),
                dbc.Col(
                    dcc.Loading(

                        dcc.Graph(id="distance-histo-graph", figure=empty_figure("Select Sequence")),
                        color="var(--bs-secondary)"

                    ),
                    width = 12,
                    style={
                        'height': '450px',  # Set maximum height
                        'overflowY': 'scroll',  # Enable vertical scroll
                    }

                ),

            ],

        ), width=12, md=6
    )
    return distance_distribution_box


def get_expected_distance_box():
    ed_box = dbc.Col(
        dbc.Card(
            [
                dbc.CardHeader(
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.H5("Expected/Median Distance"),
                                    html.Span(
                                        [
                                            html.I(className="fas fa-xl fa-question-circle fa px-2",
                                                   id="ed-tip"),
                                            dbc.Tooltip(
                                                "The upper triangular part shows the expected distance (mean) while the "
                                                "lower triangular part shows the median distance between i and j",
                                                target="ed-tip"
                                            ),
                                        ],
                                    )
                                 ], width=6, className="d-flex align-items-center"

                            ),


                        ]

                    ),

                ),
                dbc.Col(
                    dcc.Loading(

                        dcc.Graph(id="distance-heatmap-graph", figure=empty_figure("Select Sequence")),
                        color="var(--bs-secondary)"


                    ),
                    width=12,
                    style={
                        'height': '450px',  # Set maximum height
                        'overflowY': 'scroll',  # Enable vertical scroll
                    }

                ),
            ],

        ), width=12, md=6
    )
    return ed_box


def get_structures_table():
    table = dbc.Col(dbc.Card(
        [
            dbc.CardHeader(
                dbc.Row(
                    [
                        dbc.Col(html.H5("Sampled Structures"), width=6, align="center"),
                    ],
                    justify="between"
                ),

            ),
            dbc.Row(
                [

                    dbc.Col(
                        dash_table.DataTable(
                            id='structures-table',
                            columns=[
                                {"name": i, "id": i, "deletable": False, "selectable": False} for i in ["Count", "Structure"]
                            ],
                            data=None,
                            editable=True,
                            filter_action="native",
                            filter_options={"case": "insensitive"},
                            sort_action="native",
                            sort_mode="multi",
                            row_deletable=False,
                            row_selectable="multi",
                            selected_columns=[],
                            selected_rows=[],
                            page_action="native",
                            page_current=0,
                            page_size=10,
                            style_as_list_view=True,
                            style_data_conditional=[
                                {
                                    'if': {'row_index': 'odd'},
                                    'backgroundColor': 'var(--bs-secondary-bg)',
                                },
                                {
                                    'if': {'row_index': 'even'},
                                    'backgroundColor': 'var(--bs-tertiary-bg)',
                                },

                            ],
                            style_header={
                                'backgroundColor': 'var(--bs-secondary-bg)',
                                'fontWeight': 'bold',
                                "border": "none"
                            },
                            style_filter={
                                'backgroundColor': 'var(--bs-secondary-bg)',
                                'fontWeight': 'bold',
                                "border": "none !important"

                            },
                            style_data={'border': 'none !important'}
                        ),
                        width=12, style={"overflow": "auto", 'backgroundColor': 'var(--bs-primary-bg)'},
                    )

                ],
                justify="center", className="m-2"

            )

        ],
        className="shadow"



    ), width=12)
    return table

def get_forna_container():

    container = dashbio.FornaContainer(
        width=1200,
        height=600,
        id="forna-container",
        colorScheme="custom",
        customColors={},
    )

    c = dbc.Col(dbc.Card(
        [
            dbc.CardHeader(
                dbc.Row(
                    [
                        dbc.Col(html.H5("Structures"), width=6, align="center"),
                    ],
                    justify="between"
                ),

            ),
            dbc.Row(
                [

                    dbc.Col(
                        container,
                        width=12, style={"overflow": "auto", 'backgroundColor': 'var(--bs-primary-bg)'},
                    )

                ],
                justify="center", className="m-2"

            )

        ],
        className="shadow"



    ), width=12)
    return c


def get_layout():
    layout = html.Div(

        [
            dcc.Store(id="displayed-seq-hash"),
            dcc.Store(id="displayed-seq"),

            dbc.Container(
                [
                    dbc.Row(
                        get_selected_sequence()
                    ),
                    dbc.Row(
                        [
                            get_distance_distrubution_box(),
                            get_expected_distance_box()
                        ]
                    ),
                    dbc.Row(
                        get_structures_table()
                    ),
                    dbc.Row(
                        get_forna_container()
                    )
                ],

                fluid=True,

            )
        ],
        style={
            'minHeight': '85vh',  # Ensures the div takes at least the full viewport height
            'width': '100%',  # Full width
            'padding': '20px'  # Optional: adds some spacing
        },
    )
    return layout




layout = get_layout()

@callback(
    Output("seqid", "options"),
    Input("url", "pathname"),
    Input("sequence-computation-finished", "data"),
    Input("user_id", "data")


)
def update_sequence_selection(_, comp_finish, user_id):
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()

    cursor.execute("SELECT DISTINCT header FROM jobs WHERE jobs.status = ? AND jobs.user_id = ?;", ("finished", user_id))
    hashes = [row[0] for row in cursor.fetchall()]

    conn.close()
    return hashes


@callback(
    Output("nt-i", "options"),
    Output("nt-i", "value"),
    Output("nt-j", "options"),
    Output("displayed-seq-hash", "data"),
    Output("displayed-seq", "data"),
    Input("seqid", "value"),
    State("user_id", "data")
)

def update_ij_selection(header, user_id):
    if header is None:
        raise dash.exceptions.PreventUpdate
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()

    cursor.execute("""
    SELECT submissions.length, submissions.hash , submissions.sequence
FROM jobs
JOIN submissions ON jobs.hash = submissions.hash
WHERE jobs.user_id = ? AND jobs.header = ?;
""", (user_id, header))
    result = cursor.fetchone()
    print(list(result))
    d = list(range(result["length"]))
    return d, 0, d,  result["hash"].hex(), result["sequence"]


@callback(
    Output('structures-table', "data"),
    Input("displayed-seq-hash", "data"),
)
def update_structures_table(md_hash):
    rows, length = get_structures_and_length_for_hash(DATABASE_FILE, bytes.fromhex(md_hash))
    data = [{"id": row["id"],"Count": row["Count"], "Structure": bytes_to_structure(row["structure"], length)} for row in rows]
    return data


@callback(
    Output("distance-heatmap-graph", "figure", allow_duplicate=True),
    Input("displayed-seq-hash", "data"),
    State("mode-switch", "value"),
    prevent_initial_call=True


)
def plot_expected_distance(md_hash, switch):
    if md_hash is None:
        raise dash.exceptions.PreventUpdate
    md_hash = bytes.fromhex(md_hash)
    matrix = matrix_from_hash(DATABASE_FILE, md_hash)
    colorscale = [COLORS["green"], COLORS["blue"]]
    fig = expected_median_distance_maxtrix(matrix, colorscale=colorscale)
    if not switch:
        fig.update_layout(DARK_LAYOUT)
    else:
        fig.update_layout(LAYOUT)
    return fig





@callback(
    Output("distance-histo-graph", "figure", allow_duplicate=True),
    Input("displayed-seq-hash", "data"),
    Input("nt-i", "value"),
    Input("nt-j", "value"),
    State("mode-switch", "value"),
    prevent_initial_call=True


)
def plot_histo(seq_hash, i, j, switch):
    if seq_hash is None or i is None:
        return dash.no_update
    seq_hash = bytes.fromhex(seq_hash)
    matrix = matrix_from_hash(DATABASE_FILE, seq_hash)
    if j is None:
        fig = plot_distances_with_running_j(matrix, i)
    else:
        fig = distance_histo_from_matrix(matrix, i, j)

    if not switch:
        fig.update_layout(DARK_LAYOUT)
    else:
        fig.update_layout(LAYOUT)
    return fig



clientside_callback(
    ClientsideFunction(namespace='clientside', function_name='updateFornaContainer'),
    Output("forna-container", "sequences"),
    Input("structures-table", "derived_virtual_data"),
    Input("structures-table", "selected_row_ids"),
    State("displayed-seq", "data"),
)


clientside_callback(
    ClientsideFunction(namespace="clientside", function_name="switchLayout"),
    Output("distance-histo-graph", "figure"),
    Input("mode-switch", "value"),
    State("distance-histo-graph", "figure"),
    State("light-layout", "data"),
    State("dark-layout", "data")
)

clientside_callback(
    ClientsideFunction(namespace="clientside", function_name="switchLayout"),
    Output("distance-heatmap-graph", "figure"),
    Input("mode-switch", "value"),
    State("distance-heatmap-graph", "figure"),
    State("light-layout", "data"),
    State("dark-layout", "data")
)

clientside_callback(
    ClientsideFunction(
        namespace="clientside",
        function_name="highlightNucleotides"
    ),
    output=Output("forna-container", "customColors"),
    inputs=[Input("nt-i", "value"), Input("nt-j", "value")]
)