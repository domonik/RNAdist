import RNA
import dash
from dash import callback, html, clientside_callback, Input, Output, dcc, dash_table, State, Patch
import dash_bootstrap_components as dbc
from RNAdist.dashboard import DATABASE_FILE, MAX_SEQ_LENGTH
from RNAdist.sampling.ed_sampling import distance_histogram
from RNAdist.plots.sampling_plots import distance_histo_from_matrix
import sqlite3
import numpy as np
import io
from RNAdist.dashboard.helpers import hash_model_details, insert_submission, set_status, check_hash_exists, \
    get_jobs_of_user, check_user_header_combination, check_user_hash_combination, database_cleanup
import zlib
import uuid

dash.register_page(__name__, path='/submission', name="Submission")



def get_submissions_table():
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
                                {"name": i, "id": i, "deletable": False, "selectable": False} for i in ["Header", "Job ID", "Status"]
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
                            dbc.Input(id="submitted-sequence", value="AATGATGAGACCGGCACTAACTGAGTTGTGATGAGCACTCGGTTGGCTGA"),
                            width=6
                        ),
                        dbc.Col(
                            dbc.Input(id="submitted-header", value="ENA|AF254836|AF254836.1"),
                            width=6
                        ),
                        dbc.Col(
                            dbc.Button("Submit Sequence", id="submit-sequence", className="m-1"),
                            width=3, className="d-flex justify-content-end"
                        )

                    ]
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


def get_layout():
    layout = html.Div(

        [
            dcc.Store(id="submitted-job"),
            dbc.Container(
                [
                    submission_failed_modal(),
                    dbc.Row(
                        get_sequence_submission_box()
                    ),
                    dbc.Row(get_submissions_table()),
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
            'minHeight': '85vh',  # Ensures the div takes at least the full viewport height
            'width': '100%',  # Full width
            'padding': '20px'  # Optional: adds some spacing
        },
    )
    return layout


layout = get_layout()




@callback(
    Output("submitted-job", "data"),
    Output("submission-fail-modal", "is_open"),
    Output("submission-fail-text", "children"),
    Input("submit-sequence", "n_clicks"),
    State("submitted-sequence", "value"),
    State("submitted-header", "value"),
    State("user_id", "data"),
    State("submission-fail-modal", "is_open"),
    prevent_initial_call=True

)
def submit_sequence(n_clicks, sequence, header, user_id, is_open):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate
    if len(sequence) > MAX_SEQ_LENGTH:
        text = f"Your sequence is too long. Maximum allowed: {MAX_SEQ_LENGTH}\n." \
               f" You can change this when running your own RNAdist instance"
        return dash.no_update, True, text

    if check_user_header_combination(DATABASE_FILE, user_id, header):
        text = "An Entry with the header and user ID already exists. Please rename the sequence"
        return dash.no_update, True, text


    md = RNA.md()
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
    if submitted_job is None:
        return dash.no_update
    database_cleanup(db_path=DATABASE_FILE)
    header, md_dict, sequence = submitted_job
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

    return True

@callback(
    Output("submissions-table", "data"),
    Input("interval-component", "n_intervals"),
    Input("user_id", "data"),
    Input("submitted-job", "data"),
    Input("submission-fail-modal", "is_open")
)
def display_status(n, user_id, _, _2):

    status = get_jobs_of_user(DATABASE_FILE, user_id)
    table = []
    for row in status:
        job_id = str(row["hash"].hex())
        state = row["status"]
        table.append({"Status": state, "Job ID": job_id, "Header": row["header"]})
    return table