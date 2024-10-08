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
import datetime
import pathlib
from pathlib import Path
import traceback


# Run only on CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["XLA_PYTHON_CLIENT_MEM_PREALLOCATE"] = "False"
# Flag to disable JAX JIT compilation
# os.environ["JAX_DISABLE_JIT"] = "True"
# Flag to solve MUMPS hanging
os.environ["OMP_NUM_THREADS"] = "1"

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
js.load_model(
    robot_model_init, s=s_0[to_js], xyz_rpy=xyz_rpy_0, visualization_mode="record"
)
s_js, ds_js, tau_js = js.get_state()
t = 0.0
H_b = js.get_base()
w_b = js.get_base_velocity()

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

# %%
# ==== Define the simulation loop ====


def simulate(
    T: float,
    js: JaxsimSimulator,
    tsid: TSIDController,
    mpc: CentroidalMPC,
    to_mj: list[int],
    to_js: list[int],
    s_ref: list[float],
) -> dict[str, np.array]:

    # Acronyms:
    # - lf, rf: left foot, right foot
    # - js: JaxSim
    # - tsid: Task Space Inverse Dynamics
    # - mpc: Model Predictive Control
    # - sfp: Swing Foot Planner
    # - mj: Mujoco
    # - s: joint positions
    # - ds: joint velocities
    # - τ: joint torques
    # - b: base
    # - com: center of mass
    # - dcom: center of mass velocity

    # Logging
    s_js_log = []
    ds_js_log = []
    W_p_CoM_js_log = []
    W_p_lf_js_log = []
    W_p_rf_js_log = []
    W_p_CoM_mpc_log = []
    W_p_lf_sfp_log = []
    W_p_rf_sfp_log = []
    f_lf_mpc_log = []
    f_rf_mpc_log = []
    f_lf_js_log = []
    f_rf_js_log = []
    tau_tsid_log = []
    W_p_CoM_tsid_log = []
    t_log = []

    # Define number of steps
    n_step_tsid_js = int(tsid.frequency / dt)
    n_step_mpc_tsid = int(mpc.get_frequency_seconds() / tsid.frequency)
    print(f"{n_step_mpc_tsid=}, {n_step_tsid_js=}")
    counter = 0
    mpc_success = True
    succeded_controller = True

    t = 0.0

    while t < T:
        try:
            print(f"==== Time: {t:.4f}s ====", flush=True, end="\r")

            # Reading robot state from simulator
            s_js, ds_js, tau_js = js.get_state()
            H_b = js.get_base()
            w_b = js.get_base_velocity()
            t = js.get_simulation_time()

            # Update TSID
            tsid.set_state_with_base(
                s=s_js[to_mj], s_dot=ds_js[to_mj], H_b=H_b, w_b=w_b, t=t
            )

            # MPC plan
            if counter == 0:
                mpc.set_state_with_base(
                    s=s_js[to_mj], s_dot=ds_js[to_mj], H_b=H_b, w_b=w_b, t=t
                )
                mpc.update_references()
                mpc_success = mpc.plan_trajectory()
                mpc.contact_planner.advance_swing_foot_planner()
                if not (mpc_success):
                    print("MPC failed")
                    break

            # Reading new references
            com_mpc, dcom_mpc, f_lf_mpc, f_rf_mpc, ang_mom_mpc = mpc.get_references()
            lf_sfp, rf_sfp = mpc.contact_planner.get_references_swing_foot_planner()

            # f_lf_js, f_rf_js = js.get_feet_wrench()

            tsid.compute_com_position()

            # Update references TSID
            tsid.update_task_references_mpc(
                com=com_mpc,
                dcom=dcom_mpc,
                ddcom=np.zeros(3),
                left_foot_desired=lf_sfp,
                right_foot_desired=rf_sfp,
                s_desired=np.array(s_ref),
                wrenches_left=f_lf_mpc,
                wrenches_right=f_rf_mpc,
            )

            # Run control
            succeded_controller = tsid.run()

            if not (succeded_controller):
                print("Controller failed")
                break

            tau_tsid = tsid.get_torque()

            # Step the simulator
            js.step(n_step=n_step_tsid_js, torques=tau_tsid[to_js])
            counter = counter + 1

            if counter == n_step_mpc_tsid:
                counter = 0

            # Stop the simulation if the robot fell down
            if js.data.base_position()[2] < 0.5:
                print(f"Robot fell down at t={t:.4f}s.")
                break

            # Log data
            # TODO transform mpc contact forces to wrenches to be compared with jaxsim ones
            t_log.append(t)
            tau_tsid_log.append(tau_tsid)
            s_js_log.append(s_js)
            ds_js_log.append(ds_js)
            W_p_CoM_js_log.append(js.get_com_position())
            W_p_lf_js, W_p_rf_js = js.get_feet_positions()
            W_p_lf_js_log.append(W_p_lf_js)
            W_p_rf_js_log.append(W_p_rf_js)
            # f_lf_js_log.append(f_lf_js)
            # f_rf_js_log.append(f_rf_js)
            W_p_CoM_mpc_log.append(com_mpc)
            f_lf_mpc_log.append(f_lf_mpc)
            f_rf_mpc_log.append(f_rf_mpc)
            W_p_lf_sfp_log.append(lf_sfp.transform.translation())
            W_p_rf_sfp_log.append(rf_sfp.transform.translation())
            W_p_CoM_tsid_log.append(tsid.COM.toNumPy())

        except Exception as e:
            print(f"Exception during simulation at time{t}: {e}")
            traceback.print_exc()
            break

    logs = {
        "t": np.array(t_log),
        "s_js": np.array(s_js_log),
        "ds_js": np.array(ds_js_log),
        "tau_tsid": np.array(tau_tsid_log),
        "W_p_CoM_js": np.array(W_p_CoM_js_log),
        "W_p_lf_js": np.array(W_p_lf_js_log),
        "W_p_rf_js": np.array(W_p_rf_js_log),
        "f_lf_js": np.array(f_lf_js_log),
        "f_rf_js": np.array(f_rf_js_log),
        "W_p_CoM_mpc": np.array(W_p_CoM_mpc_log),
        "f_lf_mpc": np.array(f_lf_mpc_log),
        "f_rf_mpc": np.array(f_rf_mpc_log),
        "W_p_lf_sfp": np.array(W_p_lf_sfp_log),
        "W_p_rf_sfp": np.array(W_p_rf_sfp_log),
        "W_p_CoM_tsid": np.array(W_p_CoM_tsid_log),
    }

    return logs


# %%
# ==== Set simulation parameters ====

T = 10.0
dt = js.dt


# %%
# ==== Run the simulation ====


now = time.perf_counter()

logs = simulate(T=T, js=js, tsid=tsid, mpc=mpc, to_mj=to_mj, to_js=to_js, s_ref=s_0)

wall_time = time.perf_counter() - now
avg_iter_time_ms = (wall_time / (T / dt)) * 1000

print(
    f"\nRunning simulation took {wall_time:.2f}s for {T:.3f}s simulated time. \nIteration avg time of {avg_iter_time_ms:.1f} ms."
)
print(f"RTF: {T / wall_time * 100:.2f}%")

# %%
# ==== Plot the results ====

# Extract logged variables
t = logs["t"]
s_js = logs["s_js"]
ds_js = logs["ds_js"]
tau_tsid = logs["tau_tsid"]
W_p_CoM_js = logs["W_p_CoM_js"]
W_p_lf_js = logs["W_p_lf_js"]
W_p_rf_js = logs["W_p_rf_js"]
f_lf_js = logs["f_lf_js"]
f_rf_js = logs["f_rf_js"]
W_p_CoM_mpc = logs["W_p_CoM_mpc"]
f_lf_mpc = logs["f_lf_mpc"]
f_rf_mpc = logs["f_rf_mpc"]
W_p_lf_sfp = logs["W_p_lf_sfp"]
W_p_rf_sfp = logs["W_p_rf_sfp"]
W_p_CoM_tsid = logs["W_p_CoM_tsid"]

n_sim_steps = s_js.shape[0]
s_0_to_js = np.full_like(a=s_js, fill_value=s_0[to_js])

# Joint tracking
fig, axs = plt.subplots(
    nrows=int(np.ceil(len(js_joint_names) / 2)), ncols=2, sharex=True, figsize=(12, 16)
)
for idx, name in enumerate(js_joint_names):
    ax = axs[idx // 2, idx % 2]
    ax.title.set_text(name)
    ax.plot(t, s_js[:, idx] * 180 / np.pi, label="Simulated")
    ax.plot(
        t,
        s_0_to_js[:, idx] * 180 / np.pi,
        linestyle="--",
        label="Reference",
    )
    ax.grid()
    ax.set_ylabel("[deg]")
    ax.legend()
plt.suptitle("Joint tracking")
plt.show()

# Joint tracking error
fig, axs = plt.subplots(
    nrows=int(np.ceil(len(js_joint_names) / 2)), ncols=2, sharex=True, figsize=(12, 16)
)
for idx, name in enumerate(js_joint_names):
    ax = axs[idx // 2, idx % 2]
    ax.title.set_text(name)
    ax.plot(t, (s_js[:, idx] - s_0_to_js[:, idx]) * 180 / np.pi)
    ax.grid()
    ax.set_ylabel("[deg]")
plt.suptitle("Joint tracking error (reference - simulated)")
plt.tight_layout()
plt.show()

# Feet height
fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True)
ax = axs[0]
ax.title.set_text("Left foot sole height")
ax.plot(t, W_p_lf_js[:, 2], label="Simulated")
ax.plot(t, W_p_lf_sfp[:, 2], label="Swing Foot Planner reference")
ax.legend()
ax.grid()
ax.set_ylabel("Height [m]")
ax = axs[1]
ax.title.set_text("Right foot sole height")
ax.plot(t, W_p_rf_js[:, 2], label="Simulated")
ax.plot(t, W_p_rf_sfp[:, 2], label="Swing Foot Planner reference")
ax.legend()
ax.grid()
plt.show()

# COM tracking
fig = plt.figure()
ax1, ax2, ax3 = fig.subplots(nrows=3, ncols=1, sharex=True)
ax1.title.set_text("Center of mass: x component")
ax1.plot(t, W_p_CoM_js[:, 0], label="Simulated")
ax1.plot(t, W_p_CoM_mpc[:, 0], linestyle="--", label="MPC References")
ax2.title.set_text("Center of mass: y component")
ax2.plot(t, W_p_CoM_js[:, 1], label="Simulated")
ax2.plot(t, W_p_CoM_mpc[:, 1], linestyle="--", label="MPC References")
ax3.title.set_text("Center of mass: z component")
ax3.plot(t, W_p_CoM_js[:, 2], label="Simulated")
ax3.plot(t, W_p_CoM_mpc[:, 2], linestyle="--", label="MPC References")
ax1.legend()
ax2.legend()
ax3.legend()
ax1.grid()
ax2.grid()
ax3.grid()
plt.xlabel("Time [s]")
plt.show(block=False)

# Torques
fig, axs = plt.subplots(
    nrows=int(np.ceil(len(js_joint_names) / 2)), ncols=2, sharex=True, figsize=(12, 12)
)
for idx, name in enumerate(js_joint_names):
    ax = axs[idx // 2, idx % 2]
    ax.title.set_text(name)
    ax.plot(t, tau_tsid[:, idx], label="TSID References")
    ax.legend()
    ax.grid()
    ax.set_ylabel("[Nm]")
plt.suptitle("Joint torques")
plt.tight_layout()
plt.show()

# Contact forces
fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(12, 12))
titles = [
    "Left foot force - x component",
    "Right foot force - x component",
    "Left foot force - y component",
    "Right foot force - y component",
    "Left foot force - z component",
    "Right foot force - z component",
]
for row in range(3):
    for col in range(2):
        idx = row * 2 + col
        # force_js = f_lf_js[:, col] if col == 0 else f_rf_js[:, col]
        force_mpc = f_lf_mpc[:, col] if col == 0 else f_rf_mpc[:, col]
        # axs[row, col].plot(t, force_js, label="Simulated")
        axs[row, col].plot(t, force_mpc, linestyle="--", label="MPC References")
        axs[row, col].set_title(titles[idx])
        axs[row, col].set_ylabel("Force [N]")
        axs[row, col].set_xlabel("Time [s]")
        axs[row, col].grid()
        axs[row, col].legend()

plt.suptitle("Feet contact forces")
plt.tight_layout()
plt.show()


# Save video
# Create results folder if not existing
def get_repo_root(current_path: Path = Path(__file__).parent) -> Path:
    current_path = current_path.resolve()

    for parent in current_path.parents:
        if (parent / ".git").exists():
            return parent

    raise RuntimeError("No .git directory found, not a Git repository.")


def create_output_dir(directory: Path):
    # Create the directory if it doesn't exist
    directory.mkdir(parents=True, exist_ok=True)


# Usage
repo_root = get_repo_root()

# Define the results directory
results_dir = repo_root / "examples" / "results"

# Create the results directory if it doesn't exist
create_output_dir(results_dir)
now = datetime.datetime.now()
current_time = now.strftime("%Y-%m-%d_%H-%M-%S")
filepath = results_dir / pathlib.Path(current_time + "simulation_comodo.mp4")
js.save_video(filepath)
print(f"Video saved at: {filepath}")
