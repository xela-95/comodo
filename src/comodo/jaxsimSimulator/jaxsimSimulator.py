import numpy as np
import numpy.typing as npt
from comodo.abstractClasses.simulator import Simulator
from jaxsim.high_level.common import VelRepr
from jaxsim.high_level.model import Model
from jaxsim.simulation.ode import compute_contact_forces
from jaxsim.simulation.ode_data import ODEState
from typing import Union
import jax.numpy as jnp
import logging
from meshcat_viz.world import MeshcatWorld


class JaxsimSimulator(Simulator):
    def __init__(self) -> None:
        super().__init__()

    def load_model(
        self,
        robot_model,
        s=None,
        xyz_rpy: npt.ArrayLike = None,
        kv_motors=None,
        Im=None,
    ) -> None:
        logging.warning("Motor parameters are not supported in JaxsimSimulator")
        logging.warning("Defaulting to ground parameters: K=1e6, D=2e3, mu=0.5")
        xyz_rpy[2] = xyz_rpy[2] + 0.005
        self.model = Model.build_from_model_description(
            model_description=robot_model.urdf_string,
            model_name=robot_model.robot_name,
            vel_repr=VelRepr.Mixed,
            is_urdf=True,
        )
        self.model.reduce(considered_joints=robot_model.joint_name_list)
        self.model.reset_base_position(position=jnp.array(xyz_rpy[:3]))
        self.model.reset_base_orientation(
            orientation=jnp.array(self.RPY_to_quat(*xyz_rpy[3:]))
        )

        self.model.reset_joint_positions(positions=s)
        self.dt = 0.0001

        if False:
            self.renderer = MeshcatWorld()
            self.renderer.insert_model(
                model_description=robot_model.urdf_string,
                is_urdf=True,
                model_name=robot_model.robot_name,
            )
            self.renderer._visualizer.jupyter_cell()

    def get_contact_wrenches(self) -> npt.ArrayLike:
        x0 = ODEState(
            physics_model=self.model.data.model_state,
            soft_contacts=self.model.data.contact_state,
        )

        (
            contact_forces_links,
            tangential_deformation_dot,
            contact_forces_points,
        ) = compute_contact_forces(physics_model=self.model.physics_model, ode_state=x0)

        return np.array(contact_forces_links)

    def set_input(self, input: npt.ArrayLike) -> None:
        self.model.set_joint_generalized_force_targets(jnp.array(input))

    def step(self, n_step: int = 1) -> None:
        self.model.integrate(t0=0, tf=n_step * self.dt, sub_steps=1)

    def get_base(self) -> npt.ArrayLike:
        return np.array(self.model.base_transform())

    def get_base_velocity(self) -> npt.ArrayLike:
        return np.array(self.model.base_velocity())

    def get_state(self) -> Union[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:
        s = np.array(self.model.joint_positions())
        s_dot = np.array(self.model.joint_velocities())
        tau = np.array(self.model.data.model_input.tau)

        return s, s_dot, tau

    def close(self) -> None:
        pass

    def RPY_to_quat(self, roll, pitch, yaw):
        cr = np.cos(roll / 2)
        cp = np.cos(pitch / 2)
        cy = np.cos(yaw / 2)
        sr = np.sin(roll / 2)
        sp = np.sin(pitch / 2)
        sy = np.sin(yaw / 2)

        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy

        return [qw, qx, qy, qz]

    def render(self):
        self.renderer.update_model(
            model_name=self.model.name,
            joint_positions=self.model.joint_positions(),
            joint_names=self.model.joint_names(),
            base_position=self.model.base_position(),
            base_quaternion=self.model.base_orientation(),
        )
