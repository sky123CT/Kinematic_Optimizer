from casadi import *
from robotic import *
from optimizer import DirectCollocationDA, quaternion2rm, calculate_rel_rm
import numpy as np
from matplotlib import pyplot as plt
import copy

time_interval_num = 20
robot_joint_num_list = [7, 6]
iiwa_acc_limits = [5.0, 5.0, 3.0, 2.0, 2.0, 2.0, 2.0]
ur_acc_limits = [5.0, 5.0, 3.0, 2.0, 2.0, 2.0]

rel_distance_W = DM(np.diag([10] * 3))# DM(np.diag([100, 100, 1000]))
rel_orientation_W = DM(np.diag([10]))# DM(np.diag([100, 100, 100]))

movement_W1 = DM(np.diag([1] * robot_joint_num_list[0]))
max_velocity_W1 = DM(np.diag([1/1000.0] * robot_joint_num_list[0]))
input_W1 = DM(np.diag([0.1] * robot_joint_num_list[0]))

movement_W2 = DM(np.diag([1] * robot_joint_num_list[1]))
max_velocity_W2 = DM(np.diag([1/1000.0] * robot_joint_num_list[1]))
input_W2 = DM(np.diag([0.1] * robot_joint_num_list[1]))

time_W = DM(np.diag([1000000]))


robot_group = MultiRobot(arm_num=2,
                         robot_joint_num_list=[7, 6],
                         robot_file='../model_file/demo_robdekon_scanning.urdf',
                         root_link='world',
                         end_link_list=['zimmer_ee_iiwa', 'zimmer_ee_ur'])
robot_iiwa = copy.deepcopy(robot_group.robot_arm_1)
robot_ur = copy.deepcopy(robot_group.robot_arm_2)
robot_iiwa.acc_limits = iiwa_acc_limits
robot_ur.acc_limits = ur_acc_limits

robot_iiwa_q0 = robot_iiwa.generate_random_joint_angles()
robot_iiwa_qe = robot_iiwa.generate_random_joint_angles()
robot_ur_q0 = robot_ur.generate_random_joint_angles()
robot_ur_qe = robot_ur.generate_random_joint_angles()

rel_pos_target = robot_iiwa.get_position_fk()(robot_iiwa_qe)[:3, 3] - robot_ur.get_position_fk()(robot_ur_qe)[:3, 3]
rm_iiwa = quaternion2rm(robot_iiwa.get_quaternion_fk()(robot_iiwa_qe))
rm_ur = quaternion2rm(robot_ur.get_quaternion_fk()(robot_ur_qe))
rel_ori_target = calculate_rel_rm(rm_iiwa, rm_ur)

optimizer_dc = DirectCollocationDA(robot1=copy.deepcopy(robot_iiwa),
                                   robot2=copy.deepcopy(robot_ur),
                                   rel_pos_target=rel_pos_target,
                                   rel_ori_target=rel_ori_target,
                                   robot1_q0=robot_iiwa_q0,
                                   robot2_q0=robot_ur_q0,
                                   collocation_num=3,
                                   interpolation_dim_position=3,
                                   interpolation_dim_rotation=4,
                                   time_interval_num=time_interval_num,
                                   rel_distance_W=rel_distance_W,
                                   rel_orientation_W=rel_orientation_W,
                                   movement_W1=movement_W1,
                                   max_v_W1=max_velocity_W1,
                                   input_W1=input_W1,
                                   movement_W2=movement_W2,
                                   max_v_W2=max_velocity_W2,
                                   input_W2=input_W2,
                                   time_W=time_W)
x_iiwa_opt, x_iiwa_dot_opt, u_iiwa_opt, x_ur_opt, x_ur_dot_opt, u_ur_opt, t_opt, _= optimizer_dc.optimize()
print("------------------------------ Pos -------------------------------------")
print("robot_iiwa_q0:", robot_iiwa_q0)
print("robot_iiwa_qe:", robot_iiwa_qe)
print("robot_ur_q0:", robot_ur_q0)
print("robot_ur_qe:", robot_ur_qe)
print("x1_0", x_iiwa_opt[:, 0])
print("x1_e", x_iiwa_opt[:, time_interval_num])
print("x2_0", x_ur_opt[:, 0])
print("x2_e", x_ur_opt[:, time_interval_num])
print("--------------------------- Target Pos ----------------------------------")
actual_end_pos_iiwa = robot_iiwa.get_position_fk()(x_iiwa_opt[:, time_interval_num])[:3, 3]
actual_end_ori_iiwa = robot_iiwa.get_quaternion_fk()(x_iiwa_opt[:, time_interval_num])
actual_end_pos_ur = robot_ur.get_position_fk()(x_ur_opt[:, time_interval_num])[:3, 3]
actual_end_ori_ur = robot_ur.get_quaternion_fk()(x_ur_opt[:, time_interval_num])

actual_end_rm_iiwa = quaternion2rm(actual_end_ori_iiwa)
actual_end_rm_ur = quaternion2rm(actual_end_ori_ur)
actual_end_rel_ori = calculate_rel_rm(actual_end_rm_iiwa, actual_end_rm_ur)

print("Target Relative Position: ", rel_pos_target)
print("Actual End relative Position: ", actual_end_pos_iiwa - actual_end_pos_ur)
print("Target Relative Orientation: ", rel_ori_target)
print("Actual End Orientation: ", actual_end_rel_ori)
print("------------------------- Execution time --------------------------------")
time = t_opt.full()[0][0]
print("Execution time: ", time)
print("------------------------- Print all --------------------------------")
print("x_iiwa_opt:", x_iiwa_opt)
print("x_iiwa_dot_opt:", x_iiwa_dot_opt)
print("u_iiwa_opt:", u_iiwa_opt)
print("x_ur_opt:", x_ur_opt)
print("x_ur_dot_opt:", x_ur_dot_opt)
print("u_ur_opt:", u_ur_opt)


# plot
time_list = []
sum_time = 0
time_list.append(sum_time)
for t in range(time_interval_num):
    sum_time += time / time_interval_num
    time_list.append(sum_time)
plt.figure()
plt.clf()
plt.subplot(321)
for i in range(robot_joint_num_list[0]):
    plt.plot(time_list, x_iiwa_opt.full()[i, :], '-')
plt.grid()
plt.subplot(323)
for i in range(robot_joint_num_list[0]):
    plt.plot(time_list, x_iiwa_dot_opt.full()[i, :], '-')
plt.grid()
plt.subplot(325)
for i in range(robot_joint_num_list[0]):
    plt.step(time_list[:-1], u_iiwa_opt.full()[i, :], '-.')
plt.grid()
plt.subplot(322)
for i in range(robot_joint_num_list[1]):
    plt.plot(time_list, x_ur_opt.full()[i, :], '-')
plt.grid()
plt.subplot(324)
for i in range(robot_joint_num_list[1]):
    plt.plot(time_list, x_ur_dot_opt.full()[i, :], '-')
plt.grid()
plt.subplot(326)
for i in range(robot_joint_num_list[1]):
    plt.step(time_list[:-1], u_ur_opt.full()[i, :], '-.')
plt.grid()
plt.show()
