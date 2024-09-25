# This python scripts demonstrates how to use the JaxSim simulator to simulate a walking robot using comodo.

# %%
# ==== Imports ====
from __future__ import annotations
import xml.etree.ElementTree as ET
import numpy as np
import tempfile
import urllib.request
import time
import os
import matplotlib.pyplot as plt


os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["XLA_PYTHON_CLIENT_MEM_PREALLOCATE"] = "False"

from comodo.jaxsimSimulator import JaxsimSimulator
from comodo.robotModel.robotModel import RobotModel
from comodo.robotModel.createUrdf import createUrdf
from comodo.centroidalMPC.centroidalMPC import CentroidalMPC
from comodo.centroidalMPC.mpcParameterTuning import MPCParameterTuning
from comodo.TSIDController.TSIDParameterTuning import TSIDParameterTuning
from comodo.TSIDController.TSIDController import TSIDController

