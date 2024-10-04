import logging
from typing import Union
import math
import jaxsim
import jax.numpy as jnp
import jaxsim.api as js
import numpy as np
import numpy.typing as npt
from comodo.abstractClasses.simulator import Simulator
from jaxsim import VelRepr, integrators
from jaxsim.mujoco.visualizer import MujocoVisualizer
from jaxsim.mujoco.model import MujocoModelHelper
from jaxsim.mujoco.loaders import UrdfToMjcf
from jaxsim.rbda.contacts.rigid import RigidContacts, RigidContactsParams
from jaxsim.rbda.contacts.relaxed_rigid import (
    RelaxedRigidContacts,
    RelaxedRigidContactsParams,
)
from jaxsim.rbda.contacts.visco_elastic import ViscoElasticContacts
import pathlib


class JaxsimSimulator(Simulator):

    def __init__(self) -> None:
        self.dt = 0.000_5
        self.tau = jnp.zeros(20)
        self.visualize_robot_flag = None
        self.viz = None
        self.recorder = None
        self.link_contact_forces = None
        self.left_foot_link_idx = None
        self.right_foot_link_idx = None
        self.left_footsole_frame_idx = None
        self.right_footsole_frame_idx = None
        self.viz_fps = 10
        self.last_recorded_t_ns = 0.0
        self.last_rendered_t_ns = 0.0

    def load_model(
        self,
        robot_model,
        s=None,
        xyz_rpy: npt.ArrayLike = None,
        kv_motors=None,
        Im=None,
        terrain_params=None,
    ) -> None:
        logging.warning("Motor parameters are not supported in JaxsimSimulator")
        model = js.model.JaxSimModel.build_from_model_description(
            model_description=robot_model.urdf_string,
            model_name=robot_model.robot_name,
            # contact_model=RigidContacts(
            #     parameters=RigidContactsParams(mu=0.5, K=1.0e4, D=1.0e2)
            # ),
            # contact_model=RelaxedRigidContacts(
            #     parameters=RelaxedRigidContactsParams(
            #         mu=0.8,
            #         time_constant=0.005,
            #     )
            # ),
            # contact_model=ViscoElasticContacts(),
        )
        model = js.model.reduce(
            model=model,
            considered_joints=robot_model.joint_name_list,
        )

        self.data = js.data.JaxSimModelData.build(
            model=model,
            velocity_representation=VelRepr.Mixed,
            base_position=jnp.array(xyz_rpy[:3]),
            base_quaternion=jnp.array(self.RPY_to_quat(*xyz_rpy[3:])),
            joint_positions=jnp.array(s),
            contacts_params=js.contact.estimate_good_soft_contacts_parameters(
                model=model,
                number_of_active_collidable_points_steady_state=16,
                max_penetration=0.002,
            ),
        )

        self.integrator = integrators.fixed_step.RungeKutta4.build(
            dynamics=js.ode.wrap_system_dynamics_for_integration(
                model=model,
                data=self.data,
                system_dynamics=js.ode.system_dynamics,
            ),
        )

        self.integrator_state = self.integrator.init(
            x0=self.data.state, t0=0, dt=self.dt
        )

        self.model = model

        # TODO: expose these names as parameters
        self.left_foot_link_idx = js.link.name_to_idx(
            model=self.model, link_name="l_ankle_2"
        )
        self.right_foot_link_idx = js.link.name_to_idx(
            model=self.model, link_name="r_ankle_2"
        )
        self.left_footsole_frame_idx = js.frame.name_to_idx(
            model=self.model, frame_name="l_sole"
        )
        self.right_footsole_frame_idx = js.frame.name_to_idx(
            model=self.model, frame_name="r_sole"
        )

        logging.info(f"Left foot link index: {self.left_foot_link_idx}")
        logging.info(f"Right foot link index: {self.right_foot_link_idx}")
        logging.info(f"Left foot sole frame index: {self.left_footsole_frame_idx}")
        logging.info(f"Right foot sole frame index: {self.right_footsole_frame_idx}")

        mjcf_string, assets = UrdfToMjcf.convert(
            urdf=self.model.built_from,
        )

        self.mj_model_helper = MujocoModelHelper.build_from_xml(
            mjcf_description=mjcf_string, assets=assets
        )

        self.recorder = jaxsim.mujoco.MujocoVideoRecorder(
            model=self.mj_model_helper.model,
            data=self.mj_model_helper.data,
            fps=30,
        )

    def get_feet_wrench(self) -> npt.ArrayLike:
        wrenches = self.get_link_contact_forces()

        left_foot = np.array(wrenches[self.left_foot_link_idx])
        right_foot = np.array(wrenches[self.right_foot_link_idx])
        return left_foot, right_foot

    def set_input(self, input: npt.ArrayLike) -> None:
        self.tau = jnp.array(input)

    def step(self, torques: np.ndarray = None, n_step: int = 1) -> None:

        if torques is None:
            torques = np.zeros(20)

        # try:
        for _ in range(n_step):
            # self.data, self.integrator_state = jaxsim.rbda.contacts.visco_elastic.step(
            #     model=self.model,
            #     data=self.data,
            #     dt=self.dt,
            #     joint_forces=torques,
            #     link_forces=None,  # f
            # )
            self.data, self.integrator_state = js.model.step(
                model=self.model,
                data=self.data,
                dt=self.dt,
                integrator=self.integrator,
                integrator_state=self.integrator_state,
                joint_forces=torques,
                link_forces=None,  # f
            )

            current_time_ns = np.array(object=self.data.time_ns).astype(int)
            if current_time_ns - self.last_recorded_t_ns >= int(
                1e9 / self.recorder.fps
            ):
                self.record_frame()
                self.last_recorded_t_ns = current_time_ns

            if self.visualize_robot_flag and (
                current_time_ns - self.last_rendered_t_ns >= int(1e9 / self.viz_fps)
            ):
                self.render()
                self.last_rendered_t_ns = current_time_ns

        # except Exception as e:
        #     print(f"Exception in model.step:\n{e}")
        #     raise e
        # finally:
        #     self.save_video(pathlib.Path("exception_video.mp4"))

        self.link_contact_forces = js.model.link_contact_forces(
            model=self.model,
            data=self.data,
            joint_force_references=torques,
        )

    def get_base(self) -> npt.ArrayLike:
        return np.array(self.data.base_transform())

    def get_base_velocity(self) -> npt.ArrayLike:
        return np.array(self.data.base_velocity())

    def get_simulation_time(self) -> float:
        return self.data.time()

    def get_state(self) -> Union[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:
        s = np.array(self.data.joint_positions())
        s_dot = np.array(self.data.joint_velocities())
        tau = np.array(self.tau)

        return s, s_dot, tau

    def get_link_contact_forces(self) -> npt.ArrayLike:
        return self.link_contact_forces

    def total_mass(self) -> float:
        return js.model.total_mass(self.model)

    def get_feet_positions(self) -> tuple[npt.ArrayLike, npt.ArrayLike]:
        W_p_lf = js.frame.transform(
            model=self.model,
            data=self.data,
            frame_index=self.left_footsole_frame_idx,
        )[0:3, 3]
        W_p_rf = js.frame.transform(
            model=self.model,
            data=self.data,
            frame_index=self.right_footsole_frame_idx,
        )[0:3, 3]

        return (W_p_lf, W_p_rf)

    def get_com_position(self) -> npt.ArrayLike:
        return js.com.com_position(self.model, self.data)

    def close(self) -> None:
        pass

    def RPY_to_quat(self, roll, pitch, yaw):
        cr = math.cos(roll / 2)
        cp = math.cos(pitch / 2)
        cy = math.cos(yaw / 2)
        sr = math.sin(roll / 2)
        sp = math.sin(pitch / 2)
        sy = math.sin(yaw / 2)

        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy

        return [qw, qx, qy, qz]

    def render(self):
        if not self.viz:
            mjcf_string, assets = UrdfToMjcf.convert(
                urdf=self.model.built_from,
            )

            self.mj_model_helper = MujocoModelHelper.build_from_xml(
                mjcf_description=mjcf_string, assets=assets
            )

            self.viz = MujocoVisualizer(
                model=self.mj_model_helper.model, data=self.mj_model_helper.data
            )
            self._handle = self.viz.open_viewer()

        self.mj_model_helper.set_base_position(
            position=self.data.base_position(),
        )
        self.mj_model_helper.set_base_orientation(
            orientation=self.data.base_orientation(),
        )
        self.mj_model_helper.set_joint_positions(
            positions=self.data.joint_positions(),
            joint_names=self.model.joint_names(),
        )
        self.viz.sync(viewer=self._handle)

    def record_frame(self):
        self.mj_model_helper.set_base_position(
            position=self.data.base_position(),
        )
        self.mj_model_helper.set_base_orientation(
            orientation=self.data.base_orientation(),
        )
        self.mj_model_helper.set_joint_positions(
            positions=self.data.joint_positions(),
            joint_names=self.model.joint_names(),
        )

        self.recorder.record_frame()

    def save_video(self, file_path: str | pathlib.Path):
        self.recorder.write_video(path=file_path)

    def set_terrain_parameters(self, terrain_params: npt.ArrayLike) -> None:
        terrain_params_dict = dict(
            zip(["max_penetration", "damping_ratio", "mu"], terrain_params)
        )

        logging.warning(f"Setting terrain parameters: {terrain_params_dict}")

        contact_params = js.contact.estimate_good_soft_contacts_parameters(
            model=self.model,
            number_of_active_collidable_points_steady_state=16,
            max_penetration=terrain_params_dict["max_penetration"],
            damping_ratio=terrain_params_dict["damping_ratio"],
            static_friction_coefficient=terrain_params_dict["mu"],
        )

        self.data = self.data.replace(contacts_params=contact_params)
