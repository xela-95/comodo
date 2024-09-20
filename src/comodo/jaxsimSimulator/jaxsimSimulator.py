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
from jaxsim.rbda.contacts.relaxed_rigid import RelaxedRigidContacts


class JaxsimSimulator(Simulator):

    def __init__(self) -> None:
        self.dt = 1 / 1000
        self.tau = jnp.zeros(20)
        self.visualize_robot_flag = None
        self.viz = None

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
            is_urdf=True,
            contact_model=RigidContacts(
                parameters=RigidContactsParams(mu=0.5, K=1.0e4, D=1.0e2)
            ),
            # contact_model=RelaxedRigidContacts(),
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
        )

        self.integrator = integrators.fixed_step.RungeKutta4SO3.build(
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

    def get_feet_wrench(self) -> npt.ArrayLike:
        wrenches = js.model.link_contact_forces(model=self.model, data=self.data)

        left_foot = np.array(wrenches[19])
        right_foot = np.array(wrenches[20])
        return left_foot, right_foot

    def set_input(self, input: npt.ArrayLike) -> None:
        self.tau = jnp.array(input)

    def step(self, torque: np.ndarray = None, n_step: int = 1) -> None:

        if torque is None:
            torque = np.zeros(20)

        self.data, self.integrator_state = js.model.step(
            dt=self.dt,
            model=self.model,
            data=self.data,
            integrator=self.integrator,
            integrator_state=self.integrator_state,
            joint_forces=torque,
            link_forces=None,  # f
        )

        if self.visualize_robot_flag:
            self.render()

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

    def total_mass(self) -> float:
        return js.model.total_mass(self.model)

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

    def set_terrain_parameters(self, terrain_params: npt.ArrayLike) -> None:
        terrain_params_dict = dict(zip(["K", "D", "mu"], terrain_params))

        logging.warning(f"Setting terrain parameters: {terrain_params_dict}")

        self.data = self.data.replace(
            soft_contacts_params=jaxsim.rbda.SoftContactsParams.build(
                **terrain_params_dict
            )
        )
