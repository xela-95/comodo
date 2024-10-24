# CoMoDO

## Control Motion Design Optimization

Suite of parametrized controller and simulator for codesign of robots.


---

<p align="center">
  <b>⚠️ REPOSITORY UNDER DEVELOPMENT ⚠️</b>
  <br>We cannot guarantee stable API
</p>

---

## Installation

This library depends on

- [``casadi``](https://web.casadi.org/)
- [``numpy``](https://numpy.org/)
- [``idyntree``](https://github.com/robotology/idyntree)
- [``bipedal-locomotion-framework``](https://github.com/ami-iit/bipedal-locomotion-framework)
- [``adam-robotics``](https://github.com/ami-iit/ADAM)
- [``mujoco``](https://mujoco.org/)
- [``mujoco-python-viewer``](https://github.com/rohanpsingh/mujoco-python-viewer)
- [``matplotlib``](https://matplotlib.org/stable/)
- [``urllib3``](https://urllib3.readthedocs.io/en/stable/)
- [``urchin``](https://github.com/fishbotics/urchin)
- [``jaxsim``](https://github.com/ami-iit/jaxsim)

To install you can use the following commands


```
conda create -n comododev -c conda-forge adam-robotics bipedal-locomotion-framework=0.19.0 mujoco-python-viewer matplotlib urllib3 urchin notebook jaxsim

conda activate comododev
pip install --no-deps git+https://github.com/CarlottaSartore/urdf-modifiers.git@scalar_modification
pip install --no-deps -e .

```

### With hippopt 

```
conda install -c conda-forge -c robotology casadi pytest liecasadi  meshcat-python ffmpeg-python
pip install --no-deps git+https://github.com/ami-iit/hippopt.git
```
 

### With Drake 
Working with drake as the simulator backend requires the following additional dependencies,
```
conda install meshio tqdm
pip install drake git+https://github.com/ami-iit/amo_urdf
```

## Usage 

Take a look at the [examples](./examples) folder! 

### Maintainer

This repository is maintained by 
|                                                              |                                                      |
| :----------------------------------------------------------: | :--------------------------------------------------: |
| <img src="https://user-images.githubusercontent.com/56030908/135461492-6d9a1174-19bd-46b3-bee6-c4dbaea9e210.jpeg" width="40"> | [@CarlottaSartore](https://github.com/CarlottaSartore) |



