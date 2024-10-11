from casadi import *
from robotic import *
from data import H5Reader, H5Writer
from optimizer import TennisTossingOpt, quaternion2rm, calculate_rel_rm
import numpy as np
from matplotlib import pyplot as plt
import copy
import time

# global setting
if_plot = True
if_write_h5 = False
time_interval_num = 50
franka_joint_num = 7
franka_q_ul = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
franka_q_ll = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
franka_dq_l = [2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100]
franka_ddq_l = [15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0]
release_pos=DM(np.array([1.209, 0.3389, 0.7046]))
release_ori=DM(np.array([0.924, 0, -0.383, 0]))
release_vel=DM(np.array([-2.3938, 0.5396, 2.4539]))

vel_max_W = DM(np.diag([10] * franka_joint_num))
movement_W = DM(np.diag([1] * franka_joint_num))
input_W = DM(np.diag([0.1] * franka_joint_num))
time_W = DM(np.diag([1]))

robot = Robot(robot_joint_num=franka_joint_num,
              robot_file='dual_franka_panda.urdf',
              root_link='base_link',
              end_link='right_ee_link')
robot.acc_limits = franka_ddq_l

writer = H5Writer()

nlp_exec_time = np.empty((0, 1))
nlp_opt_time = np.empty((0, 1))
robot_q0 = [0, 0.1, 0, -1, 0 ,2.6, 0.9]

optimizer = TennisTossingOpt(robot=robot,
                             release_pos=release_pos,
                             release_ori=release_ori,
                             release_vel=release_vel,
                             robot_q0=robot_q0,
                             time_interval_num=time_interval_num,
                             vel_max_W=vel_max_W,
                             movement_W=movement_W,
                             input_W=input_W,
                             time_W=time_W)

x_opt, x_dot_opt, u_opt, t_opt, opt_time= optimizer.optimize()
x_opt_np = x_opt.full()
x_dot_opt_np = x_dot_opt.full()
u_opt_np = u_opt.full()
nlp_exec_time = np.concatenate((nlp_exec_time, t_opt.full()))
nlp_opt_time = np.concatenate((nlp_opt_time, np.array([opt_time])[:, np.newaxis]))

print("--------------------------- Target ----------------------------------")
actual_release_pos = robot.get_position_fk()(x_opt[:, time_interval_num/2])[:3, 3]
actual_release_ori = robot.get_quaternion_fk()(x_opt[:, time_interval_num/2])
actual_release_vel = mtimes(robot.pos_jac_mapping(x_opt[:, time_interval_num/2]), x_dot_opt[:, time_interval_num/2])
print("Target Release Position: ", release_pos.full())
print("Actual Release Position: ", actual_release_pos)
'''print("Target Release Orientation: ", release_ori.full())
print("Actual Release Orientation: ", actual_release_ori)'''
print("Target Release Velocity: ",release_vel.full())
print("Actual Release Velocity: ", actual_release_vel)

print("------------------------- Print all --------------------------------")
print("x_opt:", x_opt)
print("x_dot_opt:", x_dot_opt)
print("u_opt:", u_opt)

if if_plot:
    # plot
    time_list = []
    sum_time = 0
    time_list.append(sum_time)
    time = t_opt.full()[0][0]
    for t in range(time_interval_num):
        sum_time += time / time_interval_num
        time_list.append(sum_time)
    plt.figure()
    plt.clf()
    plt.subplot(311)
    for i in range(franka_joint_num):
        plt.plot(time_list, x_opt.full()[i, :], '-')
    plt.grid()
    plt.subplot(312)
    for i in range(franka_joint_num):
        plt.plot(time_list, x_dot_opt.full()[i, :], '-')
    plt.grid()
    plt.subplot(313)
    for i in range(franka_joint_num):
        plt.step(time_list[:-1], u_opt.full()[i, :], '-')
    plt.grid()
    plt.show()

if if_write_h5:
    write_data = {'nlp_exec_time': nlp_exec_time,
                  'nlp_opt_time': nlp_opt_time}
    writer.write_data(data=write_data)



