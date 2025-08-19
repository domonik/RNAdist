import time

import RNA
import dash
from dash import callback, html, clientside_callback, Input, Output, dcc, dash_table, State, Patch
import dash_bootstrap_components as dbc
from RNAdist.dashboard import MAX_SEQ_LENGTH, CONFIG
from RNAdist.sampling.ed_sampling import distance_histogram
from RNAdist.plots.sampling_plots import distance_histo_from_matrix
import sqlite3
import numpy as np
import io
from RNAdist.dashboard.helpers import hash_model_details, insert_submission, set_status, check_hash_exists, \
    get_jobs_of_user, check_user_header_combination, check_user_hash_combination, database_cleanup
import zlib
import uuid

DATABASE_FILE = CONFIG['DATABASE']

dash.register_page(__name__, path='/submission', name="Submission")

import logging

logger = logging.getLogger(__name__)


def get_submissions_table():
    columns = ["Header", "Job ID", "Status", "ModelDetails", "Sequence"]
    table = dbc.Col(dbc.Card(
        [
            dbc.CardHeader(
                dbc.Row(
                    [
                        dbc.Col(html.H5("Submitted Jobs"), width=6, align="center"),
                    ],
                    justify="between"
                ),

            ),
            dbc.Row(
                [

                    dbc.Col(
                        dash_table.DataTable(
                            id='submissions-table',
                            columns=[
                                {"name": i, "id": i, "deletable": False, "selectable": False} for i in columns
                            ],
                            data=None,
                            editable=True,
                            filter_action="native",
                            filter_options={"case": "insensitive"},
                            sort_action="native",
                            sort_mode="multi",
                            row_deletable=False,
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
                                {
                                    'if': {'column_id': 'Header'},
                                    'color': 'var(--bs-secondary)',   # use Bootstrap link color variable
                                    'textDecoration': 'underline',
                                    'cursor': 'pointer',
                                }

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

                            # style_cell_conditional=[
                            #     {'if': {'column_id': 'ModelDetails'}, 'maxWidth': '20%', 'overflow': 'hidden', 'textOverflow': 'ellipsis', 'whiteSpace': 'nowrap'},
                            #     {'if': {'column_id': 'Sequence'}, 'maxWidth': '20%', 'overflow': 'hidden', 'textOverflow': 'ellipsis', 'whiteSpace': 'nowrap'},
                            # ],
                            style_data={'border': 'none !important'},
                            style_table={'width': '100%', "tableLayout": "fixed", "overflowX": "auto"},
                            style_cell={
                                'minWidth': '100px', 'maxWidth': '300px',
                                'overflow': 'hidden',
                                'textOverflow': 'ellipsis',
                                'whiteSpace': 'nowrap',
                            },
                        ),

                        width=12, style={'backgroundColor': 'var(--bs-primary-bg)', "width": "100%", "overflow-y": "visible !important"},
                    )

                ],
                justify="center", className="m-2"

            )

        ],
        className="shadow"



    ), width=12)
    return table




def get_sequence_submission_box():
    submission_box = dbc.Col(
        dbc.Card(
            [
                dbc.CardHeader(
                    dbc.Row(
                        [
                            dbc.Col(html.H5("Submit Sequence"), width=6),
                        ]

                    ),

                ),
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.FormFloating(
                                [
                                    dbc.Input(id="submitted-header", value="ENA|AF254836|AF254836.1", maxlength=40),
                                    dbc.Label("Header"),

                                ]

                            ),
                            width=12, className="p-1"
                        ),
                        dbc.Col(
                            dbc.FormFloating(
                                [
                                    dbc.Textarea(
                                        id="submitted-sequence",
                                        value="AATGATGAGACCGGCACTAACTGAGTTGTGATGAGCACTCGGTTGGCTGA".replace("T", "U"),
                                        placeholder="",
                                        style={"minHeight": "7rem"}
                                    ),
                                    dbc.Label("Sequence"),

                                ]

                            ),
                            width=12, className="p-1"
                        ),

                        dbc.Col(
                            [
                                dbc.FormFloating(
                                    [
                                        dbc.Input(id="md-temperature-input", placeholder="", value=37.0, type="number", max=60, min=1, step=0.1),
                                        dbc.Label("Temperature"),
                                    ]
                                )
                            ], width=6, md=3, xl=2, className="p-1"
                        ),
                        dbc.Col(
                            [
                                dbc.FormFloating(
                                    [
                                        dbc.Input(id="md-bpspan-input", value=None, type="number", min=-1, step=1),
                                        dbc.Label("Max bp span"),
                                    ]
                                )
                            ], width=6, md=3, xl=2, className="p-1"
                        ),

                        dbc.Col(
                            dbc.Button("Submit Sequence", id="submit-sequence",),
                            width=12, className="d-flex justify-content-end p-1"
                        )

                    ],
                    className="px-5 py-4 justify-content-end"
                )


            ],
            className="shadow",
        ),
        width=12
    )
    return submission_box



def submission_failed_modal():
    modal = dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Error")),
            dbc.ModalBody(id="submission-fail-text"),
        ],
        id="submission-fail-modal",
        is_open=False,
        centered=True,
        backdrop="static",  # Prevent closing by clicking outside
    )
    return modal

def not_finished_modal():
    modal = dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Error")),
            dbc.ModalBody("Computation not finished. Wait until results are ready",),
        ],
        id="not-finished-modal",
        is_open=False,
        centered=True,
        backdrop="static",  # Prevent closing by clicking outside
    )
    return modal

def get_layout():
    layout = html.Div(

        [
            dcc.Store(id="submitted-job"),
            dbc.Container(
                [
                    submission_failed_modal(),
                    not_finished_modal(),
                    dbc.Row(
                        get_sequence_submission_box(),
                        className="py-2",
                    ),
                    dbc.Row(
                        get_submissions_table(), className="py-2",
                    ),
                    dbc.Row(
                        [
                            dcc.Interval(id="interval-component", interval=10000, n_intervals=0),
                            dbc.Col(id="status-table", width=12)

                        ]

                    )

                ],

                fluid=True,

            )
        ],
        style={
            'width': '100%',  # Full width
            #'padding': '20px'  # Optional: adds some spacing
        },
    )
    return layout


layout = get_layout()



def has_invalid_letters(seq):
    return bool(set(seq.upper()) - {'A', 'U', 'G', 'C', 'T'})


def check_valid_input(sequence, temperature, bpspan, user_id, header):
    text = None
    if has_invalid_letters(sequence):

        text = html.Div(
            [
                html.P(f"Your sequence contains non valid letters."),
                html.P("Allowed letters: A, U, G, C, T")
            ]
        )
    if len(sequence) > MAX_SEQ_LENGTH:
        text = f"Your sequence is too long. Maximum allowed: {MAX_SEQ_LENGTH}\n." \
               f" You can change this when running your own RNAdist instance"

    if check_user_header_combination(DATABASE_FILE, user_id, header):
        text = "An Entry with the header and user ID already exists. Please rename the sequence"
    return text

@callback(
    Output("submitted-job", "data"),
    Output("submission-fail-modal", "is_open"),
    Output("submission-fail-text", "children"),
    Input("submit-sequence", "n_clicks"),
    State("submitted-sequence", "value"),
    State("submitted-header", "value"),
    State("user_id", "data"),
    State("md-temperature-input", "value"),
    State("md-bpspan-input", "value"),
    State("submission-fail-modal", "is_open"),
    prevent_initial_call=True

)
def submit_sequence(n_clicks, sequence, header, user_id, temperature, max_bp_span, is_open):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate

    text = check_valid_input(sequence, temperature, max_bp_span, user_id, header)
    if text:
        return dash.no_update, 1, text

    md_dict = {
        "temperature": temperature,
        "max_bp_span": max_bp_span,
    }
    md_dict = {key: value for key, value in md_dict.items() if value is not None}

    md = RNA.md(**md_dict)
    fields, md_hash = hash_model_details(md, sequence)
    if row := check_user_hash_combination(DATABASE_FILE, user_id, md_hash):
        name = row[0]
        text = f"This Sequence was already processed using the Header: {name}"
        return dash.no_update, True, text
    status = check_hash_exists(DATABASE_FILE, md_hash)
    print(f"status: {status}")
    if status:
        set_status(DATABASE_FILE, md_hash, status, user_id, header)
        return dash.no_update, is_open, dash.no_update
    else:
        set_status(DATABASE_FILE, md_hash, "queued", user_id, header)

    return [header, fields, sequence], is_open, dash.no_update


@callback(
    output=Output("sequence-computation-finished", "data"),
    inputs=Input("submitted-job", "data"),
    state=[
        State("user_id", "data"),
    ],
    background=True,
    prevent_initial_call=True
)
def process_sequence(submitted_job, user_id):
    logger.info("STARTED long running task")

    if submitted_job is None:
        return dash.no_update
    database_cleanup(db_path=DATABASE_FILE)
    header, md_dict, sequence = submitted_job
    RNA.init_rand(42)
    md = RNA.md(**md_dict)
    fc = RNA.fold_compound(sequence, md)
    _, md_hash = hash_model_details(md, sequence)
    try:

        set_status(DATABASE_FILE, md_hash, "running", user_id, header)
        histograms, samples = distance_histogram(fc, 10000, return_samples=True)
        insert_submission(sequence, histograms, samples, fc, md, DATABASE_FILE)
        set_status(DATABASE_FILE, md_hash, "finished", user_id, header)
    except Exception as e:
        print("RUNNING")
        set_status(DATABASE_FILE, md_hash, "failed", user_id, header)
        raise e
    logger.info(f"FINISHED long running task result stored in {DATABASE_FILE}")
    return True

@callback(
    Output("submissions-table", "data"),
    Output("submissions-table", "tooltip_data"),
    Output("interval-component", "disabled"),
    Input("interval-component", "n_intervals"),
    Input("user_id", "data"),
    Input("submitted-job", "data"),
    Input("submission-fail-modal", "is_open")
)
def display_status(n, user_id, _, _2):
    status = get_jobs_of_user(DATABASE_FILE, user_id)
    table = []
    tooltips = []
    all_finished = True
    for row in status:
        job_id = str(row["hash"].hex())
        state = row["status"]
        if state != "finished":
            all_finished = False
        seq = row["sequence"] if row["sequence"] is not None else "waiting"
        model_details = str({key: row[key] for key in ["temperature", "max_bp_span"]}) if seq != "waiting" else "waiting"

        tooltips.append({
            "Status": {"value": "", "type": "text"},
            "Job ID": {"value": "", "type": "text"},
            "Header": {"value": "", "type": "text"},
            "ModelDetails": {"value": "", "type": "text"},
            "Sequence": {"value": seq, "type": "text"}  # tooltip shows full sequence on hover
        })
        row = {"Status": state, "Job ID": job_id, "Header": row["header"], "ModelDetails": model_details, "Sequence": seq}
        print(row)
        table.append(row)

    interval_disabled = all_finished

    return table, tooltips, interval_disabled


@callback(
    Output("page-wide-seqid", "data"),
    Output('url', 'pathname'),
    Output("not-finished-modal", "is_open"),
    Input('submissions-table', 'active_cell'),
    State('submissions-table', 'data'),
)
def on_row_select(active_cell, data):
    if not active_cell or active_cell["column_id"] != "Header":
        raise dash.exceptions.PreventUpdate
    row_idx = active_cell['row']
    clicked_row = data[row_idx]
    header_value = clicked_row.get('Header', None)
    status = clicked_row.get('Status', None)
    if header_value is None:
        raise dash.exceptions.PreventUpdate
    if status != "finished":
        return dash.no_update, dash.no_update, True

    return header_value, "visualization", dash.no_update