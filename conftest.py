import sys
import os

FILEP = os.path.abspath(__file__)
BASEDIR = os.path.dirname(FILEP)
CPPATH = os.path.join(BASEDIR, "CPExpectedDistance")
sys.path.append(CPPATH)

pytest_plugins = ["RNAdist.dp.tests.fixtures",
                  "RNAdist.nn.tests.data_fixtures"]


