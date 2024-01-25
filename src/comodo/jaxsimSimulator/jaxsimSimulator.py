import numpy as np
import numpy.typing as npt
from comodo.abstractClasses.simulator import Simulator
from jaxsim.high_level.common import VelRepr
from jaxsim.high_level.model import Model
from typing import Union
import jax.numpy as jnp
import logging


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
        self.dt = 0.001

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
