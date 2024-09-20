import jaxsim.api as js
import jax.numpy as jnp
from rod.builder.primitives import BoxBuilder
from jaxsim.mujoco.visualizer import MujocoVideoRecorder
from jaxsim.mujoco.loaders import RodModelToMjcf
from jaxsim.mujoco.model import MujocoModelHelper
import jaxsim

model = js.model.JaxSimModel.build_from_model_description(
    model_description=BoxBuilder(name="box", mass=1.0, x=1, y=1, z=1)
    .build_model()
    .add_link()
    .add_inertial()
    .add_visual()
    .add_collision()
    .build(),
    contact_model=jaxsim.rbda.contacts.rigid.RigidContacts(),
)

data = js.data.JaxSimModelData.build(
    model=model, velocity_representation=js.common.VelRepr.Inertial
)

data = data.reset_base_position(jnp.array([0.0, 0.0, 0.5 - 0.000_1]))

references = js.references.JaxSimModelReferences.build(
    model=model,
    data=data,
    velocity_representation=data.velocity_representation,
    link_forces=jnp.array([0.0, 0.0, -10.0, 0.0, 0.0, 0.0]),
)
mjcf_string, assets = RodModelToMjcf.convert(rod_model=model.built_from)

mj_helper = MujocoModelHelper.build_from_xml(
    mjcf_description=mjcf_string, assets=assets
)

recorder = MujocoVideoRecorder(model=mj_helper.model, data=mj_helper.data, fps=1000)

integrator = jaxsim.integrators.fixed_step.RungeKutta4SO3.build(
    dynamics=js.ode.wrap_system_dynamics_for_integration(
        model=model,
        data=data,
        system_dynamics=js.ode.system_dynamics,
    ),
)

integrator_state = integrator.init(t0=0.0, dt=0.001, x0=data.state)

W_R_B, W_P_B, W_Fc = [], [], []

for _ in range(3000):
    data, integrator_state = js.model.step(
        model=model,
        data=data,
        link_forces=references.link_forces(),
        integrator_state=integrator_state,
        integrator=integrator,
        dt=0.001,
    )

    W_R_B.append(data.base_orientation(dcm=False))
    W_P_B.append(data.base_position())
    W_Fc.append(js.model.link_contact_forces(model=model, data=data))
    print(f"Simulated: {_}/3000", end="\r")

print("Simulation done. Rendering video...")
for i in range(3000):
    mj_helper.set_base_orientation(W_R_B[i])
    mj_helper.set_base_position(W_P_B[i])

    recorder.record_frame()

import pathlib

print("Video recorded.")
recorder.write_video(pathlib.Path("box1.mp4"))

import matplotlib.pyplot as plt
import numpy as np


plt.plot(np.arange(3000) * 0.001, np.array(W_Fc)[:, 0, 2])
plt.plot(np.arange(3000) * 0.001, np.array(W_P_B)[:, 0, 2])
plt.grid(True)
plt.xlabel("Time [s]")
plt.ylabel("Contact Forces [N]")
plt.title("Contact Forces")
plt.show()
