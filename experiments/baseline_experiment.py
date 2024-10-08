from casadi import *
from robotic import *
import numpy as np
from matplotlib import pyplot as plt
import copy

time_interval_num = 20
robot_joint_num = 7
distance_W = DM(np.diag([10]*3))# DM(np.diag([100, 100, 1000]))
orientation_W = DM(np.diag([10]*4))# DM(np.diag([100, 100, 100]))
movement_W = DM(np.diag([100]*robot_joint_num))
input_W = DM(np.diag([1] * robot_joint_num))
time_W = DM(np.diag([100]))
T_as_variable = True

if T_as_variable:
    from optimizer import DirectCollocationT as DirectCollocation
else:
    from optimizer import DirectCollocation


robot = Robot(robot_joint_num=robot_joint_num)
q0 = robot.generate_random_joint_angles()
qe = robot.generate_random_joint_angles()
# qe = [-0.7456322916982949, 0.018873417798711545, -2.873635364513463, -2.068114454310836, 0.858715856998963, 0.3810065988894261, 1.613481051023887]
pos_target = robot.get_position_fk()(qe)[:3, 3]
ori_target = robot.get_quaternion_fk()(qe)

trajectory = TrajectoryPolynomials()
optimizer_dc = DirectCollocation(robot=copy.deepcopy(robot),
                                 trajectory_polynomials=copy.deepcopy(trajectory),
                                 pos_target=pos_target,
                                 ori_target=ori_target,
                                 q0=q0,
                                 time_interval_num=time_interval_num,
                                 distance_W=distance_W,
                                 orientation_W=orientation_W,
                                 movement_W=movement_W,
                                 input_W=input_W,
                                 time_W=time_W)
x_opt, xc_opt, x_dot_opt, u_opt, t_opt = optimizer_dc.optimize()
print("------------------------------ Pos -------------------------------------")
print("q0:", q0)
print("qe:", qe)
print("x0", x_opt[:, 0])
print("xe", x_opt[:, time_interval_num])
print("--------------------------- Target Pos ----------------------------------")
actual_end_pos = robot.get_position_fk()(x_opt[:, time_interval_num])[:3, 3]
actual_end_ori = robot.get_quaternion_fk()(x_opt[:, time_interval_num])
print("Target Position: ", pos_target)
print("Actual End Position: ", actual_end_pos)
print("Target Orientation: ", ori_target)
print("Actual End Orientation: ", actual_end_ori)
print("------------------------- Execution time --------------------------------")
if T_as_variable:
    time = t_opt.full()[0][0]
else:
    time = np.sum(t_opt.full())
print("Execution time: ", time)
print("------------------------- Print all --------------------------------")
print("x_opt:", x_opt)
print("x_dot_opt:", x_dot_opt)
print("u_opt:", u_opt)


# plot
time_list = []
sum_time = 0
time_list.append(sum_time)
for t in range(time_interval_num):
    if T_as_variable:
        sum_time += time / time_interval_num
    else:
        sum_time += t_opt.full()[t][0]
    time_list.append(sum_time)
plt.figure()
plt.clf()
plt.subplot(311)
for i in range(robot_joint_num):
    plt.plot(time_list, x_opt.full()[i, :], '-')
plt.subplot(312)
for i in range(robot_joint_num):
    plt.plot(time_list, x_dot_opt.full()[i, :], '-')
plt.subplot(313)
for i in range(robot_joint_num):
    plt.step(time_list[:-1], u_opt.full()[i, :], '-.')
plt.grid()
plt.show()

plt.show()


