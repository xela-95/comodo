# This script demonstrates the use of Optuna to optimize the hyper-parameter of the Jaxsim contact model.

# %%
# ==== Imports ====

from pathlib import Path
import optuna
from optuna.trial import TrialState
import xml.etree.ElementTree as ET
import numpy as np
import tempfile
import urllib.request
import jax.numpy as jnp
import os
from rich import logging
import traceback
import jax

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["XLA_PYTHON_CLIENT_MEM_PREALLOCATE"] = "False"

from comodo.robotModel.createUrdf import createUrdf
from comodo.jaxsimSimulator import JaxsimSimulator
from comodo.robotModel.robotModel import RobotModel
from comodo.centroidalMPC.centroidalMPC import CentroidalMPC
from comodo.centroidalMPC.mpcParameterTuning import MPCParameterTuning
from comodo.TSIDController.TSIDParameterTuning import TSIDParameterTuning
from comodo.TSIDController.TSIDController import TSIDController

# %%
# ==== Define functions ====


def init():
    jax.config.update("jax_platform_name", "cpu")

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

    s_0, xyz_rpy_0, H_b_0 = robot_model_init.compute_desired_position_walking()
    print(
        f"Initial configuration:\nBase position: {xyz_rpy_0[:3]}\nBase orientation: {xyz_rpy_0[3:]}\nJoint positions: {s_0}"
    )

    # Define simulator and set initial position
    js = JaxsimSimulator()
    js.load_model(robot_model_init, s=s_0[to_js], xyz_rpy=xyz_rpy_0)

    print(f"Contact model in use: {js.model.contact_model}")
    print(f"Link names:\n{js.model.link_names()}")
    print(f"Frame names:\n{js.model.frame_names()}")
    print(f"Mass: {js.total_mass()*js.data.standard_gravity()} N")

    s_js, ds_js, tau_js = js.get_state()
    return (js, s_0, xyz_rpy_0, H_b_0, robot_model_init, js_joint_names, to_mj, to_js)


def plot_study(study: optuna.Study):
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

    repo_root = get_repo_root()
    plots_dir = repo_root / "plots"

    # Create the results directory if it doesn't exist
    create_output_dir(plots_dir)

    # Plot the optimization history
    optuna.visualization.plot_optimization_history(study).write_image(
        "./plots/optimization_history.png"
    )

    # Plot the parallel coordinate
    optuna.visualization.plot_parallel_coordinate(study).write_image(
        "./plots/parallel_coordinate.png"
    )

    # Plot the parameter importance
    optuna.visualization.plot_param_importances(study).write_image(
        "./plots/param_importance.png"
    )

    # Plot loss distributions
    optuna.visualization.plot_intermediate_values(study).write_image(
        "./plots/intermediate_values.png"
    )


def simulate(
    T: float,
    js: JaxsimSimulator,
    tsid: TSIDController,
    mpc: CentroidalMPC,
    to_mj: list[int],
    to_js: list[int],
    s_ref: list[float],
) -> float:
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
    n_step_tsid_js = int(tsid.frequency / js.dt)
    n_step_mpc_tsid = int(mpc.get_frequency_seconds() / tsid.frequency)
    print(f"{n_step_mpc_tsid=}, {n_step_tsid_js=}")
    counter = 0
    mpc_success = True
    succeded_controller = True

    t = 0.0
    obj = 0.0

    while t < T:
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

        f_lf_js, f_rf_js = js.get_feet_wrench()

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

        # Stop the simulation if the controller failed
        if not (succeded_controller):
            print("Controller failed")
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
        f_lf_js_log.append(f_lf_js)
        f_rf_js_log.append(f_rf_js)
        W_p_CoM_mpc_log.append(com_mpc)
        f_lf_mpc_log.append(f_lf_mpc)
        f_rf_mpc_log.append(f_rf_mpc)
        W_p_lf_sfp_log.append(lf_sfp.transform.translation())
        W_p_rf_sfp_log.append(rf_sfp.transform.translation())
        W_p_CoM_tsid_log.append(tsid.COM.toNumPy())

        # Get a score on the controller and choose to prune the trial if it is not good
        # obj = 0.0
        # trial.report(obj, t)

        # if trial.should_prune():
        #     raise optuna.TrialPruned()

        obj = t / T

    return obj


def objective(trial: optuna.Trial) -> float:
    # Define the parameters to optimize and get the values for the trial
    max_penetration = trial.suggest_float("max_penetration", 1e-4, 1e-2)
    # damping_ratio = trial.suggest_float("damping_ratio", 0.001, 10)
    damping_ratio = 1.0
    mu = trial.suggest_float("mu", 0.25, 1.5)

    TERRAIN_PARAMETERS = (max_penetration, damping_ratio, mu)

    print(
        f"Terrain parameters: max_penetration={max_penetration}, damping_ratio={damping_ratio}, mu={mu}"
    )

    # Setup the simulation

    # s_0, xyz_rpy_0, H_b_0 = robot_model_init.compute_desired_position_walking()

    print(
        f"Initial configuration:\nBase position: {xyz_rpy_0[:3]}\nBase orientation: {xyz_rpy_0[3:]}\nJoint positions: {s_0}"
    )

    obj = 0.0
    try:
        # Set the terrain parameters
        js.set_terrain_parameters(TERRAIN_PARAMETERS)

        # Reset simulation state
        js.data = js.data.reset_base_position(
            base_position=jnp.array(xyz_rpy_0[:3]),
        )
        js.data = js.data.reset_base_quaternion(
            base_quaternion=jnp.array(js.RPY_to_quat(*xyz_rpy_0[3:])),
        )
        js.data = js.data.reset_joint_positions(
            positions=jnp.array(s_0),
        )
        js.data = js.data.reset_joint_velocities(
            velocities=jnp.zeros_like(s_0),
        )
        js.data = js.data.reset_base_velocity(
            base_velocity=jnp.zeros(6),
        )

        s_js, ds_js, tau_js = js.get_state()
        t = 0.0
        H_b = js.get_base()
        w_b = js.get_base_velocity()

        # Specify if open an interactive window to visualize the robot during the simulation
        js.visualize_robot_flag = False

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

        # MPC
        mpc.set_state_with_base(
            s=s_js[to_mj], s_dot=ds_js[to_mj], H_b=H_b, w_b=w_b, t=t
        )
        mpc.initialize_centroidal_integrator(
            s=s_js[to_mj], s_dot=ds_js[to_mj], H_b=H_b, w_b=w_b, t=t
        )
        mpc_output = mpc.plan_trajectory()

        # Launch the simulation
        obj = simulate(
            T=T, js=js, tsid=tsid, mpc=mpc, to_mj=to_mj, to_js=to_js, s_ref=s_0
        )
    except Exception as e:
        print(f"Exception in model.step:\n{e}")
        traceback.print_exc()

    return obj


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--trials", type=int, default=10)
    args = parser.parse_args()

    global js, s_0, xyz_rpy_0, H_b_0, robot_model_init, js_joint_names, to_mj, to_js, T

    js, s_0, xyz_rpy_0, H_b_0, robot_model_init, js_joint_names, to_mj, to_js = init()
    T = 1.0

    study = optuna.create_study(
        direction="maximize",
        study_name="Jaxsim Contact model tuning",
        sampler=optuna.samplers.CmaEsSampler(),
    )
    study.optimize(
        func=objective, n_trials=args.trials, show_progress_bar=True, n_jobs=args.jobs
    )

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    plot_study(study=study)
