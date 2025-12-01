"""Package containing asset and sensor configurations."""
import os

# Conveniences to other module directories via relative paths
DIFFLAB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
DIFFLAB_ASSETS_ARTICULATIONS_DIR = os.path.join(DIFFLAB_DIR, "datasets/articulations")
# print(DIFFLAB_ASSETS_ARTICULATIONS_DIR)
##
# Configuration for different assets.
##
from .quadcopter import *
