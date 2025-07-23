import os
import yaml


DASHDIR = os.path.dirname(os.path.abspath(__file__))

CONFIG_FILE = os.path.join(DASHDIR, "dashConfig.yaml")
assert os.path.exists(CONFIG_FILE), "Config file does not exist"
with open(CONFIG_FILE, "r") as handle:
    CONFIG = yaml.load(handle, Loader=yaml.SafeLoader)

EXTERNAL_CONFIG_FILE = os.getenv('RNADIST_CONFIG_FILE')
if EXTERNAL_CONFIG_FILE:
    with open(EXTERNAL_CONFIG_FILE, "r") as handle:
        EXTERNEL_CONFIG = yaml.load(handle, Loader=yaml.SafeLoader)

    CONFIG.update(EXTERNEL_CONFIG)

CACHE = None
DATABASE_FILE = None
MAX_SEQ_LENGTH = 500



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

}
DARK_LAYOUT = {
    "template": "plotly_white",
    'paper_bgcolor': 'rgba(0,0,0,0)',
    'plot_bgcolor': 'rgba(0,0,0,0)',
    "font": {"color": "white", "size": 16},
}

COMMON_LAYOUT = {
    "xaxis": {"showline": True, "mirror": True, },
    "yaxis": {"showline": True, "mirror": True, },
}