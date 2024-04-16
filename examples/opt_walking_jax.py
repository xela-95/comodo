import optuna
from optuna.samplers import CmaEsSampler
from comodo.robotModel.createUrdf import createUrdf
from comodo.jaxsimSimulator import JaxsimSimulator
from comodo.robotModel.robotModel import RobotModel
from comodo.centroidalMPC.centroidalMPC import CentroidalMPC
from comodo.centroidalMPC.mpcParameterTuning import MPCParameterTuning
from comodo.TSIDController.TSIDParameterTuning import TSIDParameterTuning
from comodo.TSIDController.TSIDController import TSIDController
from optuna.trial import TrialState
import xml.etree.ElementTree as ET
import numpy as np
import tempfile
import urllib.request

import os


def plot_study(study: optuna.Study):
    os.makedirs("./plots", exist_ok=True)

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


def init(TERRAIN_PARAMETERS):

    terrain_params = dict(zip(["K", "D", "mu"], TERRAIN_PARAMETERS))
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
    legs_link_names = ["hip_3", "lower_leg"]
    joint_name_list = [
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
    modifications = {}
    for item in legs_link_names:
        left_leg_item = "l_" + item
        right_leg_item = "r_" + item
        modifications.update({left_leg_item: 1.2})
        modifications.update({right_leg_item: 1.2})

    create_urdf_instance.modify_lengths(modifications)
    urdf_robot_string = create_urdf_instance.write_urdf_to_file()
    create_urdf_instance.reset_modifications()
    robot_model_init = RobotModel(urdf_robot_string, "stickBot", joint_name_list)
    s_des, xyz_rpy, H_b = robot_model_init.compute_desired_position_walking()

    # Define simulator and set initial position
    jax_instance = JaxsimSimulator()
    jax_instance.load_model(
        robot_model_init, s=s_des, xyz_rpy=xyz_rpy, terrain_params=terrain_params
    )
    s, ds, tau = jax_instance.get_state()
    t = 0.0  # jax_instance.get_simulation_time()
    H_b = jax_instance.get_base()
    w_b = jax_instance.get_base_velocity()

    # Define the controller parameters  and instantiate the controller
    # Controller Parameters
    tsid_parameter = TSIDParameterTuning()
    mpc_parameters = MPCParameterTuning()

    # TSID Instance
    TSID_controller_instance = TSIDController(
        frequency=0.001, robot_model=robot_model_init
    )
    TSID_controller_instance.define_tasks(tsid_parameter)
    TSID_controller_instance.set_state_with_base(s, ds, H_b, w_b, t)

    # MPC Instance
    step_lenght = 0.1
    mpc = CentroidalMPC(robot_model=robot_model_init, step_length=step_lenght)
    mpc.intialize_mpc(mpc_parameters=mpc_parameters)

    # Set desired quantities
    mpc.configure(s_init=s_des, H_b_init=H_b)
    TSID_controller_instance.compute_com_position()
    mpc.define_test_com_traj(TSID_controller_instance.COM.toNumPy())

    # Set initial robot state  and plan trajectories
    jax_instance.step(n_step=1)

    # Reading the state
    s, ds, tau = jax_instance.get_state()
    H_b = jax_instance.get_base()
    w_b = jax_instance.get_base_velocity()
    t = 0.0

    # MPC
    mpc.set_state_with_base(s=s, s_dot=ds, H_b=H_b, w_b=w_b, t=t)
    mpc.initialize_centroidal_integrator(s=s, s_dot=ds, H_b=H_b, w_b=w_b, t=t)
    mpc_output = mpc.plan_trajectory()

    # Set loop variables
    TIME_TH = 20

    # Define number of steps
    n_step = int(TSID_controller_instance.frequency / jax_instance.dt)
    n_step_mpc_tsid = int(
        mpc.get_frequency_seconds() / TSID_controller_instance.frequency
    )

    counter = 0
    mpc_success = True
    energy_tot = 0.0
    succeded_controller = True

    mj_list = [
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
    get_joint_map = lambda from_, to: np.array(list(map(to.index, from_)))
    joint_map = get_joint_map(mj_list, jax_instance.model.joint_names())
    assert all(np.array(mj_list) == np.array(joint_name_list)[joint_map])

    return (
        jax_instance,
        s_des,
        TSID_controller_instance,
        mpc,
        n_step,
        n_step_mpc_tsid,
        counter,
        t,
    )


def objective(trial):
    K = trial.suggest_float("K", 1e3, 1e7, log=True)
    D = trial.suggest_float("D", 1e2, 1e4, log=True)
    mu = trial.suggest_float("mu", 0.0, 1.0)

    TERRAIN_PARAMETERS = (K, D, mu)

    (
        jax_instance,
        s_des,
        TSID_controller_instance,
        mpc,
        n_step,
        n_step_mpc_tsid,
        counter,
        t,
    ) = init(TERRAIN_PARAMETERS)

    # contact_forces = []
    # Simulation-control loop
    while t < 10:
        t = t + jax_instance.dt

        # Reading robot state from simulator
        s, ds, tau = jax_instance.get_state()

        H_b = jax_instance.get_base()
        w_b = jax_instance.get_base_velocity()

        # Update TSID
        TSID_controller_instance.set_state_with_base(
            s=s, s_dot=ds, H_b=H_b, w_b=w_b, t=t
        )

        # MPC plan
        if counter == 0:
            mpc.set_state_with_base(s=s, s_dot=ds, H_b=H_b, w_b=w_b, t=t)
            mpc.update_references()
            mpc_success = mpc.plan_trajectory()
            mpc.contact_planner.advance_swing_foot_planner()
            if not (mpc_success):
                print("MPC failed")
                break

        # Reading new references
        com, dcom, forces_left, forces_right = mpc.get_references()
        left_foot, right_foot = mpc.contact_planner.get_references_swing_foot_planner()

        left_foot_force, right_foot_force = jax_instance.get_feet_wrench()
        # contact_forces.append([left_foot_force, right_foot_force])

        # Update references TSID
        TSID_controller_instance.update_task_references_mpc(
            com=com,
            dcom=dcom,
            ddcom=np.zeros(3),
            left_foot_desired=left_foot,
            right_foot_desired=right_foot,
            s_desired=np.array(s_des),
            wrenches_left=forces_left,
            wrenches_right=forces_right,
        )

        # Run control
        succeded_controller = TSID_controller_instance.run()

        # Get a score on the controller
        score = np.square(
            left_foot_force[:3] - forces_left + right_foot_force[:3] - forces_right
        ).sum()

        trial.report(score, t)

        if not (succeded_controller):
            print("Controller failed")
            break

        tau = TSID_controller_instance.get_torque()

        jax_instance.set_input(tau)
        jax_instance.step(n_step=n_step)
        counter = counter + 1

        if counter == n_step_mpc_tsid:
            counter = 0

    return t


if __name__ == "__main__":
    import argparse

    args = argparse.ArgumentParser()
    args.add_argument("--jobs", type=int, default=1)

    study = optuna.create_study(
        direction="maximize", study_name="MPC-TSID", sampler=CmaEsSampler()
    )
    study.optimize(
        objective, n_trials=100, show_progress_bar=True, n_jobs=args.parse_args().jobs
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
