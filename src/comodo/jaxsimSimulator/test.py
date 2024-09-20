import jax.numpy as jnp
import jaxsim.api as js
import rod.builder.primitives
import rod.urdf.exporter
from jaxsim import integrators

# Create on-the-fly a ROD model of a box.
rod_model = (
    rod.builder.primitives.BoxBuilder(x=0.3, y=0.2, z=0.1, mass=1.0, name="box")
    .build_model()
    .add_link()
    .add_inertial()
    .add_visual()
    .add_collision()
    .build()
)

# Export the URDF string.
urdf_string = rod.urdf.exporter.UrdfExporter.sdf_to_urdf_string(
    sdf=rod_model, pretty=True
)

model1 = js.model.JaxSimModel.build_from_model_description(
    model_description=urdf_string,
    is_urdf=True,
)

model2 = js.model.JaxSimModel.build_from_model_description(
    model_description=urdf_string,
    is_urdf=True,
)

# Build the data
data1 = js.data.JaxSimModelData.build(model=model1)

data2 = js.data.JaxSimModelData.build(model=model2)

# Create the integrators
integrator1 = integrators.fixed_step.Heun2SO3.build(
    dynamics=js.ode.wrap_system_dynamics_for_integration(
        model=model1,
        data=data1,
        system_dynamics=js.ode.system_dynamics,
    ),
)

integrator2 = integrators.fixed_step.Heun2SO3.build(
    dynamics=js.ode.wrap_system_dynamics_for_integration(
        model=model2,
        data=data2,
        system_dynamics=js.ode.system_dynamics,
    ),
)

# ! Try to initialize the integrator
integrator_state1 = integrator1.init(x0=data1.state, t0=0, dt=1e-3)

integrator_state2 = integrator2.init(x0=data2.state, t0=0, dt=1e-3)
