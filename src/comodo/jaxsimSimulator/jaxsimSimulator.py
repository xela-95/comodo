import logging
from typing import Union

import jax
import jax.numpy as jnp
import jaxsim.api as js
import numpy as np
import numpy.typing as npt
from comodo.abstractClasses.simulator import Simulator
from jaxsim import VelRepr, integrators
from jaxsim.high_level.common import VelRepr
from jaxsim.high_level.model import Model
from jaxsim.physics.algos.soft_contacts import SoftContactsParams
from jaxsim.simulation.ode import compute_contact_forces
from jaxsim.simulation.ode_data import ODEState
from jaxsim.simulation.ode_integration import IntegratorType
from jaxsim import sixd

# from meshcat_viz.world import MeshcatWorld


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
        model = js.model.JaxSimModel.build_from_model_description(
            model_description=robot_model.urdf_string,
            model_name=robot_model.robot_name,
            is_urdf=True,
        )
        self.model = js.model.reduce(
            model=model, considered_joints=robot_model.joint_name_list
        )

        data0 = js.data.JaxSimModelData.build(
            model=self.model,
            velocity_representation=VelRepr.Inertial,
            base_position=jnp.array(xyz_rpy[:3]),
            base_quaternion=jnp.array(self.RPY_to_quat(*xyz_rpy[3:])),
            joint_positions=jnp.array(s),
        )

        self.data = data0.replace(
            soft_contacts_params=js.contact.estimate_good_soft_contacts_parameters(
                model, number_of_active_collidable_points_steady_state=2
            ),
        )

        self.integrator = integrators.fixed_step.RungeKutta4SO3.build(
            dynamics=js.ode.wrap_system_dynamics_for_integration(
                model=self.model,
                data=self.data,
                system_dynamics=js.ode.system_dynamics,
            ),
        )

        self.dt = 1e-4
        self.tau = jnp.zeros(20)

        self.integrator_state = self.integrator.init(
            x0=self.data.state, t0=0, dt=self.dt
        )

        if False:
            self.renderer = MeshcatWorld()
            self.renderer.insert_model(
                model_description=robot_model.urdf_string,
                is_urdf=True,
                model_name=robot_model.robot_name,
            )
            self.renderer._visualizer.jupyter_cell()

    def get_feet_wrench(self) -> npt.ArrayLike:
        wrenches = self.data.aux["tf"]["contact_forces_links"]

        left_foot = np.array(wrenches[-2])
        right_foot = np.array(wrenches[-1])
        return left_foot, right_foot

    def set_input(self, input: npt.ArrayLike) -> None:
        self.tau = jnp.array(input)

    def step(
        self, n_step: int = 1, terrain_parameters: tuple[float, float, float] = None
    ) -> None:
        if terrain_parameters is not None:
            K, D, mu = terrain_parameters
        self.data, self.integrator_state = js.model.step(
            dt=self.dt,
            model=self.model,
            data=self.data,
            integrator=self.integrator,
            integrator_state=self.integrator_state,
            joint_forces=self.tau,
        )

    def get_base(self) -> npt.ArrayLike:
        base_position = np.vstack(self.data.state.physics_model.base_position)

        base_unit_quaternion = (
            self.data.state.physics_model.base_quaternion.squeeze()
            / jnp.linalg.norm(self.data.state.physics_model.base_quaternion)
        )

        # wxyz -> xyzw
        to_xyzw = np.array([1, 2, 3, 0])

        base_orientation = sixd.so3.SO3.from_quaternion_xyzw(
            base_unit_quaternion[to_xyzw]
        ).as_matrix()

        return np.vstack(
            [
                np.block([base_orientation, base_position]),
                np.array([0, 0, 0, 1]),
            ]
        )

    def get_base_velocity(self) -> npt.ArrayLike:
        return np.array(self.data.base_velocity())

    def get_state(self) -> Union[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:
        s = np.array(self.data.state.physics_model.joint_positions)
        s_dot = np.array(self.data.state.physics_model.joint_velocities)
        tau = np.array(self.tau)

        return s, s_dot, tau

    def total_mass(self) -> float:
        return js.model.total_mass(self.model)

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
