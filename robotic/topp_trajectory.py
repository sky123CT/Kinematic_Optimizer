import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import tracikpy
import toppra as ta
import toppra.constraint as constraint
import toppra.algorithm as algo
import h5py
from scipy.spatial.transform import Rotation as R
from concurrent.futures import ThreadPoolExecutor
from joblib import Parallel, delayed


# Simulation parameters
Kp = 2
dt = 0.1
N_LINKS = 7  # Franka has 7 degrees of freedom
N_ITERATIONS = 1000000
NUM_THREADS = 32  # Number of threads to use

# States
WAIT_FOR_NEW_GOAL = 1
MOVING_TO_GOAL = 2

show_animation = True

if show_animation:
    plt.ion()

# Define the DH parameters for the Franka Emika Panda robot
# These parameters should be replaced with the actual DH parameters of your robot
a = [0, 0, 0.0825, -0.0825, 0, 0.088, 0]
d = [0.333, 0, 0.316, 0, 0.384, 0, 0.107]
alpha = [-np.pi/2, np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, np.pi/2, 0]

def dh_transform(a, d, alpha, theta):
    """ Compute the DH transformation matrix. """
    return np.array([[np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
                     [np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
                     [0, np.sin(alpha), np.cos(alpha), d],
                     [0, 0, 0, 1]])

def forward_kinematics(joint_angles):
    """ Compute the forward kinematics for the robot arm. """
    T = np.eye(4)
    positions = [T[:3, 3]]

    for i in range(N_LINKS):
        T = T @ dh_transform(a[i], d[i], alpha[i], joint_angles[i])
        positions.append(T[:3, 3])

    return positions

def transformation_matrix_to_pos_quat(matrix):
    """ Convert transformation matrix to position and quaternion. """
    pos = matrix[:3, 3]
    rot = R.from_matrix(matrix[:3, :3])
    quat = rot.as_quat()  # Returns [x, y, z, w]
    return pos, quat

def generate_random_joint_angles():
    # Define the joint limits for Franka Emika Panda robot (in radians)
    joint_limits = {
        'joint1': (-2.8973, 2.8973),
        'joint2': (-1.7628, 1.7628),
        'joint3': (-2.8973, 2.8973),
        'joint4': (-3.0718, -0.0698),
        'joint5': (-2.8973, 2.8973),
        'joint6': (-0.0175, 3.7525),
        'joint7': (-2.8973, 2.8973)
    }

    # Generate random angles within the specified limits
    random_joint_angles = [
        np.random.uniform(joint_limits['joint1'][0], joint_limits['joint1'][1]),
        np.random.uniform(joint_limits['joint2'][0], joint_limits['joint2'][1]),
        np.random.uniform(joint_limits['joint3'][0], joint_limits['joint3'][1]),
        np.random.uniform(joint_limits['joint4'][0], joint_limits['joint4'][1]),
        np.random.uniform(joint_limits['joint5'][0], joint_limits['joint5'][1]),
        np.random.uniform(joint_limits['joint6'][0], joint_limits['joint6'][1]),
        np.random.uniform(joint_limits['joint7'][0], joint_limits['joint7'][1])
    ]

    return random_joint_angles

def get_random_pose_6d():
    from random import uniform

    # Define the workspace limits for the Franka robot
    # Position limits (in meters)
    x_min, x_max = -0.855, 0.855
    y_min, y_max = -0.855, 0.855
    z_min, z_max = 0.0, 1.19

    # Orientation limits (in radians)
    roll_min, roll_max = -3.14, 3.14
    pitch_min, pitch_max = -3.14, 3.14
    yaw_min, yaw_max = -3.14, 3.14

    return [
        uniform(x_min, x_max),
        uniform(y_min, y_max),
        uniform(z_min, z_max),
        uniform(roll_min, roll_max),
        uniform(pitch_min, pitch_max),
        uniform(yaw_min, yaw_max)
    ]

def euler_to_rotation_matrix(roll, pitch, yaw):
    # Compute individual rotation matrices
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])

    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])

    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])

    # Combine rotations
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

def get_transformation_matrix(goal_6d):
    # Create the rotation matrix
    R = euler_to_rotation_matrix(goal_6d[3], goal_6d[4], goal_6d[5])

    # Create the transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [goal_6d[0], goal_6d[1], goal_6d[2]]

    return T

def distance_to_goal(current_pos, goal_pos):
    x_diff = goal_pos[0] - current_pos[0]
    y_diff = goal_pos[1] - current_pos[1]
    z_diff = goal_pos[2] - current_pos[2]
    return np.array([x_diff, y_diff, z_diff]).T, np.linalg.norm([x_diff, y_diff, z_diff])

def ang_diff(theta1, theta2):
    """
    Returns the difference between two angles in the range -pi to +pi
    """
    return (theta1 - theta2 + np.pi) % (2 * np.pi) - np.pi

def toppra_solver(way_pts):
    ss = np.linspace(0, 1, 2)

    # velocity and acceleration constraints
    vlims = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100])
    alims = np.array([15, 7.5, 10, 12.5, 15, 20, 20])

    path = ta.SplineInterpolator(ss, way_pts)
    pc_vel = constraint.JointVelocityConstraint(vlims)
    pc_acc = constraint.JointAccelerationConstraint(alims)

    instance = algo.TOPPRA([pc_vel, pc_acc], path, parametrizer="ParametrizeConstAccel")
    jnt_traj = instance.compute_trajectory()

    ts_sample = np.linspace(0, jnt_traj.duration, 100)
    qs_sample = jnt_traj(ts_sample)
    qds_sample = jnt_traj(ts_sample, 1)
    qdds_sample = jnt_traj(ts_sample, 2)

    return jnt_traj

def generate_data(i_goal):
    urdf_path = "franka.urdf"  # Replace with the actual path to the Franka URDF
    base_link = "panda_link0"
    end_effector_link = "panda_hand"

    ik_solver = tracikpy.TracIKSolver(urdf_path, base_link, end_effector_link)

    joint_angles = generate_random_joint_angles()
    random_start = ik_solver.fk(joint_angles)
    joint_goal_angles = generate_random_joint_angles()
    random_goal = ik_solver.fk(joint_goal_angles)

    joint_dist = np.abs(np.array(joint_goal_angles) - np.array(joint_angles))
    max_joint_dist = np.max(joint_dist)
    max_joint_id = np.argmax(joint_dist)
    # print("maximum joint angle diff: ", max_joint_dist, max_joint_id)

    start_pose = ik_solver.fk(joint_angles)
    goal_pos = ik_solver.fk(joint_goal_angles)

    errors, distance = distance_to_goal(start_pose, goal_pos)
    # print("Euclidean distance: ", distance)

    if joint_goal_angles is not None:
        tg_traj = toppra_solver(np.array([joint_angles, joint_goal_angles]))

        gridpoints = np.linspace(0, tg_traj.duration, 100)
        velocities = tg_traj(gridpoints, 1)

        vlims = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100])

        max_times = np.max(np.abs(velocities) / np.array(vlims), axis=0)
        slowest_joint_index = np.argmax(max_times)
        if slowest_joint_index == max_joint_id:
            pnt_shape_label = 1
            # print(f"The slowest joint index ({slowest_joint_index}) is the same as max_joint_id ({max_joint_id}).")
        else:
            pnt_shape_label = 0
            # print(f"The slowest joint index ({slowest_joint_index}) is different from max_joint_id ({max_joint_id}).")

        # print("execution time: ", tg_traj.duration)
        execution_time = tg_traj.duration

        # Convert transformation matrices to position and quaternion
        start_pos, start_quat = transformation_matrix_to_pos_quat(random_start)
        goal_pos, goal_quat = transformation_matrix_to_pos_quat(random_goal)

        # Save the h5
        return {
            "joint_angles": joint_angles,
            "joint_goal_angles": joint_goal_angles,
            "duration": execution_time,
            "start_pos": start_pos,
            "start_quat": start_quat,
            "goal_pos": goal_pos,
            "goal_quat": goal_quat
        }
    else:
        return None

def save_data_to_h5(data, h5f, i_goal):
    if data:
        grp = h5f.create_group(f"goal_{i_goal}")
        grp.create_dataset("joint_angles", data=data["joint_angles"])
        grp.create_dataset("joint_goal_angles", data=data["joint_goal_angles"])
        grp.create_dataset("duration", data=data["duration"])
        grp.create_dataset("start_pos", data=data["start_pos"])
        grp.create_dataset("start_quat", data=data["start_quat"])
        grp.create_dataset("goal_pos", data=data["goal_pos"])
        grp.create_dataset("goal_quat", data=data["goal_quat"])

def franka_robot():
    # HDF5 file to save the h5
    with h5py.File('franka_robot_data.h5', 'w') as h5f:
        # Generate h5 in parallel using joblib
        results = Parallel(n_jobs=NUM_THREADS)(delayed(generate_data)(i_goal) for i_goal in range(N_ITERATIONS))
        for i_goal, data in enumerate(results):
            print(i_goal)
            save_data_to_h5(data, h5f, i_goal)

        # with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        #     futures = [executor.submit(generate_data, i_goal) for i_goal in range(N_ITERATIONS)]
        #     for i_goal, future in enumerate(futures):
        #         print(i_goal)
        #         h5 = future.result()
        #         save_data_to_h5(h5, h5f, i_goal)

def tracik_main():
    franka_robot()

if _name_ == '_main_':
    tracik_main()