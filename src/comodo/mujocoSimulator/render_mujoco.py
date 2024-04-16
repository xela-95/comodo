import mujoco
import math
import numpy as np
import mujoco_viewer
import jaxsim


class MujocoVisualizer:
    def __init__(self) -> None:
        pass

    def load_from_mujoco_path(self, robot_model, s, xyz_rpy, mujoco_path):
        self.robot_model = robot_model
        self.model = mujoco.MjModel.from_xml_path(mujoco_path)
        self.data = mujoco.MjData(self.model)
        self.set_joint_vector_in_mujoco(s)
        self.set_base_pose_in_mujoco(xyz_rpy=xyz_rpy)
        mujoco.mj_forward(self.model, self.data)
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)

    def load_model(self, robot_model, s, xyz_rpy, kv_motors=None, Im=None):
        self.robot_model = robot_model
        mujoco_xml = robot_model.get_mujoco_model()
        self.model = mujoco.MjModel.from_xml_string(mujoco_xml)
        self.data = mujoco.MjData(self.model)
        self.set_joint_vector_in_mujoco(s)
        self.set_base_pose_in_mujoco(xyz_rpy=xyz_rpy)
        mujoco.mj_forward(self.model, self.data)
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)

    def set_base_pose_in_mujoco(self, xyz_rpy):
        base_xyz_quat = np.zeros(7)
        base_xyz_quat[:3] = xyz_rpy[:3]
        base_xyz_quat[3:] = self.RPY_to_quat(xyz_rpy[3], xyz_rpy[4], xyz_rpy[5])
        base_xyz_quat[2] = base_xyz_quat[2]
        self.data.qpos[:7] = base_xyz_quat

    def set_joint_vector_in_mujoco(self, pos):
        indexes_joint = self.model.jnt_qposadr[1:]
        for i in range(self.robot_model.NDoF):
            self.data.qpos[indexes_joint[i]] = pos[i]

    def update_vis(self, s, H_b):
        xyz_rpy = self.xyz_rpy_from_matrix(H_b)
        self.set_base_pose_in_mujoco(xyz_rpy)
        self.set_joint_vector_in_mujoco(s)
        mujoco.mj_forward(self.model, self.data)
        self.viewer.render()

    def close(self):
        self.viewer.close()

    def visualize_robot(self):
        self.viewer.render()

    def get_simulation_time(self):
        return self.data.time

    def get_simulation_frequency(self):
        return self.model.opt.timestep

    def xyz_rpy_from_matrix(self, matrix):
        # Extract translation (position) from the transformation matrix
        position = matrix[:3, 3]

        # Extract rotation matrix from the transformation matrix
        rotation_matrix = matrix[:3, :3]

        # Convert rotation matrix to euler angles (roll, pitch, yaw)
        # Using atan2 to handle numerical stability
        pitch = np.arctan2(
            -rotation_matrix[2, 0],
            np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2),
        )
        if np.abs(np.cos(pitch)) > 1e-6:
            roll = np.arctan2(
                rotation_matrix[2, 1] / np.cos(pitch),
                rotation_matrix[2, 2] / np.cos(pitch),
            )
            yaw = np.arctan2(
                rotation_matrix[1, 0] / np.cos(pitch),
                rotation_matrix[0, 0] / np.cos(pitch),
            )
        else:
            # Gimbal lock case
            roll = 0.0
            yaw = np.arctan2(rotation_matrix[1, 2], rotation_matrix[0, 2])

        return np.asarray([*position, roll, pitch, yaw])

    def close_visualization(self):
        if self.visualize_robot_flag:
            self.viewer.close()

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
