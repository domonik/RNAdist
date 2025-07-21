
import os
import dash
from dash import Dash, html, dcc, clientside_callback, Input, Output, State, ALL, ClientsideFunction, DiskcacheManager
import dash_bootstrap_components as dbc
from RNAdist.dashboard import DATABASE_FILE, LAYOUT, DARK_LAYOUT
import RNAdist.dashboard
import diskcache
import uuid

FILEDIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(FILEDIR, "assets")
assert os.path.exists(ASSETS_DIR)


dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"

print("Defining APP")

CACHE = diskcache.Cache(DATABASE_FILE + ".callbacks")
RNAdist.dashboard.CACHE = CACHE
background_callback_manager = DiskcacheManager(CACHE)

app = Dash(
    __name__,
    external_stylesheets=["custom.css", dbc.icons.FONT_AWESOME, dbc_css],
    assets_folder=ASSETS_DIR,
    prevent_initial_callbacks='initial_duplicate',
    use_pages=True,
    background_callback_manager=background_callback_manager

)
print("APP defined")


color_mode_switch = html.Span(
    [
        dbc.Label(className="fa fa-xl fa-moon", html_for="switch", style={"vertical-align": "0 !important"}),
        dbc.Switch( id="mode-switch", value=True, className="d-inline-block ms-3 fs-4", persistence=True),
        dbc.Label(className="fa fa-xl fa-sun", html_for="switch", style={"vertical-align": "0 !important"}),
    ]
)


# Initialize status store if not exists
if "job_status" not in CACHE:
    CACHE["job_status"] = {}



def get_navbar():
    navbar = dbc.Navbar(
        dbc.Container(
            [
                dcc.Location(id='url', refresh="callback-nav"),

                html.A(
                    dbc.Row(
                        [

                            dbc.Col(html.Img(
                                src=app.get_asset_url("BioInfLogo.png"),
                                height="50px"),
                            ),
                            dbc.Col(dbc.NavbarBrand("RNAdist Explorer", className="ms-2"), className="d-flex align-items-center"),
                        ]

                    )

                ),
                dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),

                dbc.Collapse(
                    [
                        dbc.NavItem(dbc.NavLink(f"{page['name']}", href=page["relative_path"], id={"type": f"nav-item", "index": idx}), className="p-1") for
                        idx, page in enumerate(dash.page_registry.values())
                    ],
                    is_open=False,
                    navbar=True,
                    id="navbar-collapse2",

                ),
                dbc.Collapse(
                    [
                        dbc.Col(dbc.Input(id="displayed_user_id"), width=5, className="px-2"),
                        color_mode_switch
                    ] + [
                        dbc.NavItem(
                            html.Img(
                                src="https://cd.uni-freiburg.de/wp-content/uploads/2022/09/ufr-logo-white-2.png",
                                height="50px"),
                            className="ps-0 ps-md-3"

                        )
                    ],
                    id="navbar-collapse",
                    className="justify-content-end",
                    is_open=False,
                    navbar=True,
                    ),




            ],
            fluid=True,
            className="dbc text-light"

        ),
        dark=True,
        color="var(--bs-ufr-navbar)",
        className="ufr-navbar shadow w-100", style={"position": "fixed", "z-index": "99999"}
    )
    return navbar


def _get_footer():
    div = html.Footer(
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(width=3),
                        dbc.Col(width=3),
                        dbc.Col(
                            html.Ul(
                                [
                                    html.Li(html.A(html.I(className="fa-brands fa-2xl fa-github"), target="_blank", href="https://github.com/domonik/RNAdist", className="text-light")),
                                    html.Li(html.A(html.I(className="fa-solid fa-2xl fa-envelope"), target="_blank", href=f"mailto:me", className="text-light")),
                                ],
                                className="icon-list d-flex align-items-end justify-content-end text-light"
                            ),
                            width=3,
                            align="end"
                        ),
                    ],
                    className="w-100 py-4", justify="between"
                )
            ],
            fluid=True
        ),
        className="ufr-navbar text-light mt-4", style={"z-index": "20"}
    )
    return div

def get_layout():
    layout = html.Div(
        [
            get_navbar(),
            dcc.Store(id="user_id", storage_type="session"),
            dcc.Store(id="sequence-computation-finished"),
            dcc.Store(id="light-layout", data=LAYOUT),
            dcc.Store(id="dark-layout", data=DARK_LAYOUT),

            dbc.Container(
                dash.page_container,
                fluid=True,
                style={"padding-top": "65px"}

            ),
            _get_footer()
        ], className="dbc text-light"
    )
    return layout


@app.callback(
    Output("user_id", "data"),
    Output("displayed_user_id", "value"),
    Input("user_id", "data"),
    Input("displayed_user_id", "value")
)
def assign_user_id(user_id, displayed_id):
    print(user_id, displayed_id)
    if displayed_id:
        return displayed_id, displayed_id
    if user_id is not None:
        return user_id, user_id
    uid = str(uuid.uuid4())
    return uid, uid


@app.callback(
    Output("navbar-collapse", "is_open"),
    Output("navbar-collapse2", "is_open"),
    [Input("navbar-toggler", "n_clicks")],
    [
        State("navbar-collapse", "is_open"),
    ],
)
def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open, not is_open
    return is_open, is_open


@app.callback(
    Output({'type': 'nav-item', 'index': ALL}, 'active'),
    Input("url", "pathname"),
    State({'type': 'nav-item', 'index': ALL}, 'href'),
)
def update_active_nav(url, state):
    d = [url == href for href in state]
    return d


clientside_callback(
    """
    (switchOn) => {
       document.documentElement.setAttribute("data-bs-theme", switchOn ? "light" : "dark"); 
       return window.dash_clientside.no_update
    }
    """,
    Output("mode-switch", "id"),
    Input("mode-switch", "value"),
)




