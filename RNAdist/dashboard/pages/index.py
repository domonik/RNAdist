
from dash import dcc, html
import dash_bootstrap_components as dbc
import os
import dash

dash.register_page(__name__, path='/', name="Welcome")


md_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "index.md")

with open(md_file) as handle:
    text = handle.read()

def welcome_layout(text):
    welcome = [

        dbc.Col(
            [
                dbc.Card(
                    [
                        html.Div(
                            [
                                dcc.Markdown(text, dangerously_allow_html=True, ),
                            ]
                        )
                    ]
                    , className="shadow p-2", style={"font-size": "20px"})

            ],
            width=12, lg=6, className="py-1"
        )
    ]

    return welcome

    return welcome

layout = html.Div(
    welcome_layout(text), className="row p-1 justify-content-around",
    style={
        'width': '100%',  # Full width
        'padding': '20px'  # Optional: adds some spacing
    },
)