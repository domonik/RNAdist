import sys
import os
import pytest

FILEP = os.path.abspath(__file__)
BASEDIR = os.path.dirname(FILEP)
CPPATH = os.path.join(BASEDIR, "CPExpectedDistance")
sys.path.append(CPPATH)

pytest_plugins = [
    "RNAdist.dp.tests.fixtures",
]


def pytest_addoption(parser):
    parser.addoption(
        "--skipslow", action="store_true", default=False, help="skip slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--skipslow"):
        skip_slow = pytest.mark.skip(reason="--skipslow option prevents run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)



