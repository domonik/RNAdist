
CACHE = None
DATABASE_FILE = None
MAX_SEQ_LENGTH = 250



COLORS = {
    "blue": "#344A9A",
    "green": "#00a082",
    "pink": "#f5c2ed"
}

LAYOUT = {
    "template": "plotly_white",
    'paper_bgcolor': 'rgba(0,0,0,0)',
    'plot_bgcolor': 'rgb(219, 219, 219)',
    "font": {"color": "black", "size": 16},
    "margin": {"b": 10, "t": 10},

}
DARK_LAYOUT = {
    "template": "plotly_white",
    'paper_bgcolor': 'rgba(0,0,0,0)',
    'plot_bgcolor': 'rgba(0,0,0,0)',
    "font": {"color": "white", "size": 16},
    "margin": {"b": 10, "t": 10}
}

COMMON_LAYOUT = {
    "xaxis": {"showline": True, "mirror": True, "linecolor": "white"},
    "yaxis": {"showline": True, "mirror": True, "linecolor": "white"},
}