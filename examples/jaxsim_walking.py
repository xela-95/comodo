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

# %%
# ==== Load the stickbot model ====

# Getting stickbot urdf file and convert it to string
urdf_robot_file = tempfile.NamedTemporaryFile(mode="w+")
url = "https://raw.githubusercontent.com/icub-tech-iit/ergocub-gazebo-simulations/master/models/stickBot/model.urdf"
urllib.request.urlretrieve(url, urdf_robot_file.name)
# Load the URDF file
tree = ET.parse(urdf_robot_file.name)
root = tree.getroot()

# Convert the XML tree to a string
robot_urdf_string_original = ET.tostring(root)

create_urdf_instance = createUrdf(
    original_urdf_path=urdf_robot_file.name, save_gazebo_plugin=False
)

js_joint_names = [
    "l_hip_pitch",  # 0
    "l_shoulder_pitch",  # 1
    "r_hip_pitch",  # 2
    "r_shoulder_pitch",  # 3
    "l_hip_roll",  # 4
    "l_shoulder_roll",  # 5
    "r_hip_roll",  # 6
    "r_shoulder_roll",  # 7
    "l_hip_yaw",  # 8
    "l_shoulder_yaw",  # 9
    "r_hip_yaw",  # 10
    "r_shoulder_yaw",  # 11
    "l_knee",  # 12
    "l_elbow",  # 13
    "r_knee",  # 14
    "r_elbow",  # 15
    "l_ankle_pitch",  # 16
    "r_ankle_pitch",  # 17
    "l_ankle_roll",  # 18
    "r_ankle_roll",  # 19
]
mj_joint_names = [
    "r_shoulder_pitch",  # 0
    "r_shoulder_roll",  # 1
    "r_shoulder_yaw",  # 2
    "r_elbow",  # 3
    "l_shoulder_pitch",  # 4
    "l_shoulder_roll",  # 5
    "l_shoulder_yaw",  # 6
    "l_elbow",  # 7
    "r_hip_pitch",  # 8
    "r_hip_roll",  # 9
    "r_hip_yaw",  # 10
    "r_knee",  # 11
    "r_ankle_pitch",  # 12
    "r_ankle_roll",  # 13
    "l_hip_pitch",  # 14
    "l_hip_roll",  # 15
    "l_hip_yaw",  # 16
    "l_knee",  # 17
    "l_ankle_pitch",  # 18
    "l_ankle_roll",  # 19
]

# Check that joint list from mujoco and jaxsim have the same elements (just ordered differently)
get_joint_map = lambda from_, to: np.array(list(map(to.index, from_)))
to_mj = get_joint_map(mj_joint_names, js_joint_names)
to_js = get_joint_map(js_joint_names, mj_joint_names)

assert np.array_equal(np.array(js_joint_names)[to_mj], mj_joint_names)
assert np.array_equal(np.array(mj_joint_names)[to_js], js_joint_names)

urdf_robot_string = create_urdf_instance.write_urdf_to_file()
robot_model_init = RobotModel(urdf_robot_string, "stickBot", mj_joint_names)

# %%
# ==== Compute initial configuration ====

s_0, xyz_rpy_0, H_b_0 = robot_model_init.compute_desired_position_walking()
# Update base height
# xyz_rpy_0[2] = xyz_rpy_0[2]  # + 0.007

# Override s_0
# s_0 = np.array(
#     [
#         0.0,
#         0.251,
#         0.0,
#         0.616,
#         0.0,
#         0.251,
#         0.0,
#         0.616,
#         0.50082726,
#         0.00300592,
#         -0.00164537,
#         -1.0,
#         -0.49917522,
#         -0.00342677,
#         0.49922533,
#         0.00300592,
#         -0.00163911,
#         -1.0,
#         -0.50077713,
#         -0.00342377,
#     ]
# )

print(
    f"Initial configuration:\nBase position: {xyz_rpy_0[:3]}\nBase orientation: {xyz_rpy_0[3:]}\nJoint positions: {s_0}"
)

# %%
# ==== Define JaxSim simulator and set initial position ====

js = JaxsimSimulator()
js.load_model(robot_model_init, s=s_0[to_js], xyz_rpy=xyz_rpy_0)
s_js, ds_js, tau_js = js.get_state()
t = 0.0
H_b = js.get_base()
w_b = js.get_base_velocity()
js.visualize_robot_flag = True
js.render()

print(f"Contact model in use: {js.model.contact_model}")
print(f"Link names:\n{js.model.link_names()}")
print(f"Frame names:\n{js.model.frame_names()}")
print(f"Mass: {js.total_mass()*js.data.standard_gravity()} N")

# %%
# ==== Define the controller parameters  and instantiate the controller ====

# Controller Parameters
tsid_parameter = TSIDParameterTuning()
mpc_parameters = MPCParameterTuning()

# TSID Instance
tsid = TSIDController(frequency=0.01, robot_model=robot_model_init)
tsid.define_tasks(tsid_parameter)
tsid.set_state_with_base(s_js[to_mj], ds_js[to_mj], H_b, w_b, t)

# MPC Instance
step_length = 0.1
mpc = CentroidalMPC(robot_model=robot_model_init, step_length=step_length)
mpc.intialize_mpc(mpc_parameters=mpc_parameters)

# Set desired quantities
mpc.configure(s_init=s_0, H_b_init=H_b_0)
tsid.compute_com_position()
mpc.define_test_com_traj(tsid.COM.toNumPy())

# Set initial robot state  and plan trajectories
js.step()

# Reading the state
s_js, ds_js, tau_js = js.get_state()
H_b = js.get_base()
w_b = js.get_base_velocity()
t = 0.0

# MPC
mpc.set_state_with_base(s=s_js[to_mj], s_dot=ds_js[to_mj], H_b=H_b, w_b=w_b, t=t)
mpc.initialize_centroidal_integrator(
    s=s_js[to_mj], s_dot=ds_js[to_mj], H_b=H_b, w_b=w_b, t=t
)
mpc_output = mpc.plan_trajectory()

