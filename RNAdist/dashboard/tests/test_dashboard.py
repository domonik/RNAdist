import pytest
import yaml
from RNAdist.dashboard import CONFIG_FILE

def test_default_config():
    with open(CONFIG_FILE, "r") as handle:
        CONFIG = yaml.load(handle, Loader=yaml.SafeLoader)
    assert CONFIG['DATABASE']['type'] == "sqlite" # This ensures i dont push code testing stuff to main
