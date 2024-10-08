from casadi import *
from .utility import *
import numpy as np
from robotic import *
import copy
import time


class DirectCollocationDA:
    def __init__(self,
                 robot1:Robot,
                 robot2:Robot,
                 rel_pos_target,
                 rel_ori_target,
                 robot1_q0,
                 robot2_q0,
                 collocation_num=3,
                 interpolation_dim_position=3,
                 interpolation_dim_rotation=4,
                 time_interval_num=10,
                 rel_distance_W=np.array([1, 1, 1]),
                 rel_orientation_W=np.array([1]),
                 movement_W1=np.array([1]*7),
                 max_v_W1=np.array([1]*7),
                 input_W1=np.array([1]*7),
                 movement_W2=np.array([1] * 7),
                 max_v_W2=np.array([1] * 7),
                 input_W2=np.array([1] * 7),
                 time_W=np.array([1]*10)):

        self._collocation_num = collocation_num
        self._collocation_roots = np.append(0, collocation_points(self._collocation_num, 'legendre'))
        self._interpolation_dim_position = interpolation_dim_position
        self._interpolation_dim_rotation = interpolation_dim_rotation

        self._time_interval_num = time_interval_num

        #define variable
        self._variable= self.__define_variables__(robot_joint_dim1=robot1.get_joint_num(),
                                                  robot_joint_dim2=robot2.get_joint_num())

        #define obj function
        self.objective_function = self.__define_obj_function__(
            robot1=copy.deepcopy(robot1),
            robot2=copy.deepcopy(robot2),
            rel_pos_target=rel_pos_target,
            rel_ori_target=rel_ori_target,
            robot1_q0=robot1_q0,
            robot2_q0=robot2_q0,
            rel_dist_W=rel_distance_W,
            rel_ori_W=rel_orientation_W,
            movement_W1=movement_W1,
            max_v_W1=max_v_W1,
            input_W1=input_W1,
            movement_W2=movement_W2,
            max_v_W2=max_v_W2,
            input_W2=input_W2,
            time_W=time_W)

        #define constraints
        self.constraints = self.__define_constraints__(
            robot1=copy.deepcopy(robot1),
            robot2=copy.deepcopy(robot2),
            rel_pos_target=rel_pos_target,
            rel_ori_target=rel_ori_target,
            robot1_q0=robot1_q0,
            robot2_q0=robot2_q0)

    def optimize(self):
        L = self.objective_function
        w = self.constraints['w']
        lbw = self.constraints['lbw']
        ubw = self.constraints['ubw']
        w0 = self.constraints['w0']
        g = self.constraints['g']
        lbg = self.constraints['lbg']
        ubg = self.constraints['ubg']
        x1_opt = self.constraints['x1_opt']
        x1_dot_opt = self.constraints['x1_dot_opt']
        u1_opt = self.constraints['u1_opt']
        x2_opt = self.constraints['x2_opt']
        x2_dot_opt = self.constraints['x2_dot_opt']
        u2_opt = self.constraints['u2_opt']
        t_opt = self.constraints['t_opt']

        w = vertcat(*w)
        g = vertcat(*g)
        x1_opt = horzcat(*x1_opt)
        x1_dot_opt = horzcat(*x1_dot_opt)
        u1_opt = horzcat(*u1_opt)
        x2_opt = horzcat(*x2_opt)
        x2_dot_opt = horzcat(*x2_dot_opt)
        u2_opt = horzcat(*u2_opt)
        t_opt = horzcat(*t_opt)
        trajectories = Function('trajectories',
                                [w],
                                [x1_opt, x1_dot_opt, u1_opt, x2_opt, x2_dot_opt, u2_opt, t_opt],
                                ['w'],
                                ['x1', 'x1_dot', 'u1', 'x2', 'x2_dot', 'u2', 't'])


        # Create an NLP solver
        prob = {'f': L, 'x': w, 'g': g}
        opti_setting = {
            'ipopt.max_iter': 200,
            'ipopt.print_level': 0,
            'print_time': 0
        }
        solver = nlpsol('solver', 'ipopt', prob, opti_setting)

        # Solve the NLP
        start = time.time()
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        end = time.time()
        optimization_time = end - start
        x1_opt, x1_dot_opt, u1_opt, x2_opt, x2_dot_opt, u2_opt, t_opt = trajectories(sol['x'])

        return x1_opt, x1_dot_opt, u1_opt, x2_opt, x2_dot_opt, u2_opt, t_opt, optimization_time


    def __define_variables__(self,
                             robot_joint_dim1,
                             robot_joint_dim2):
        # time
        time = MX.sym('T')

        # robot1 link theta as state
        q1 = []
        for i in range(self._time_interval_num + 1):
            q1i = []
            for j in range(robot_joint_dim1):
                q1i.append(MX.sym('q1' + str(j) + '_at_' + str(i)))
            q1.append(vertcat(*q1i))
        q1 = horzcat(*q1)

        # velocity
        q1_dot = []
        for i in range(self._time_interval_num+1):
            q1_dot_i = []
            for j in range(robot_joint_dim1):
                q1_dot_i.append(MX.sym('q1_dot_' + str(j) + '_at_' + str(i)))
            q1_dot.append(vertcat(*q1_dot_i))
        q1_dot = horzcat(*q1_dot)

        # control input variable
        q1_ddot = []
        for i in range(self._time_interval_num):
            q1_ddot_i = []
            for j in range(robot_joint_dim1):
                q1_ddot_i.append(MX.sym('q1_ddot_' + str(j) + '_at_' + str(i)))
            q1_ddot.append(vertcat(*q1_ddot_i))
        q1_ddot = horzcat(*q1_ddot)

        # robot2 link theta as state
        q2 = []
        for i in range(self._time_interval_num + 1):
            q2i = []
            for j in range(robot_joint_dim2):
                q2i.append(MX.sym('q2' + str(j) + '_at_' + str(i)))
            q2.append(vertcat(*q2i))
        q2 = horzcat(*q2)

        # velocity
        q2_dot = []
        for i in range(self._time_interval_num + 1):
            q2_dot_i = []
            for j in range(robot_joint_dim2):
                q2_dot_i.append(MX.sym('q2_dot_' + str(j) + '_at_' + str(i)))
            q2_dot.append(vertcat(*q2_dot_i))
        q2_dot = horzcat(*q2_dot)

        # control input variable
        q2_ddot = []
        for i in range(self._time_interval_num):
            q2_ddot_i = []
            for j in range(robot_joint_dim2):
                q2_ddot_i.append(MX.sym('q2_ddot_' + str(j) + '_at_' + str(i)))
            q2_ddot.append(vertcat(*q2_ddot_i))
        q2_ddot = horzcat(*q2_ddot)

        variable = {'time': time,
                    'q1': q1,
                    'q1_dot': q1_dot,
                    'q1_ddot': q1_ddot,
                    'q2': q2,
                    'q2_dot': q2_dot,
                    'q2_ddot': q2_ddot}

        return variable


    def __define_obj_function__(self,
                                robot1:Robot,
                                robot2:Robot,
                                rel_pos_target,
                                rel_ori_target,
                                robot1_q0,
                                robot2_q0,
                                rel_dist_W,
                                rel_ori_W,
                                movement_W1,
                                max_v_W1,
                                input_W1,
                                movement_W2,
                                max_v_W2,
                                input_W2,
                                time_W):
        # unpack variable
        time = self._variable['time']
        q1 = self._variable['q1']
        q1_dot = self._variable['q1_dot']
        q1_ddot = self._variable['q1_ddot']
        q2 = self._variable['q2']
        q2_dot = self._variable['q2_dot']
        q2_ddot = self._variable['q2_ddot']

        # relative quaternion orientation and cartesian relative distance
        T_fk1 = robot1.get_position_fk()
        T_fk2 = robot2.get_position_fk()
        rel_distance_cost = 0

        q_fk1 = robot1.get_quaternion_fk()
        q_fk2 = robot2.get_quaternion_fk()
        rel_orientation_cost = 0
        for t in range(self._time_interval_num):
            tcp_orientation1 = q_fk1(q1[:, t])
            tcp_orientation2 = q_fk2(q2[:, t])
            rm1 = quaternion2rm(tcp_orientation1)
            rm2 = quaternion2rm(tcp_orientation2)
            rel_rm = calculate_rel_rm(rm1, rm2)

            tcp_position1 = T_fk1(q1[:, t])[:3, 3]
            tcp_position2 = T_fk2(q2[:, t])[:3, 3]
            rel_position = mtimes(rm1.T, (tcp_position2 - tcp_position1))

            rel_orientation_cost += rel_ori_W * compare_2_rm(rel_rm, rel_ori_target)
            rel_distance_cost += ((rel_position - rel_pos_target).T @
                                  rel_dist_W @
                                  (rel_position - rel_pos_target))


        #velocity maximizing cost
        max_velocity_cost1 = 0
        for t in range(1, self._time_interval_num):
            max_velocity_cost1 += (q1_dot[:, t].T @
                                   max_v_W1 @
                                   q1_dot[:, t])

        max_velocity_cost2 = 0
        for t in range(1, self._time_interval_num):
            max_velocity_cost2 += (q2_dot[:, t].T @
                                   max_v_W2 @
                                   q2_dot[:, t])
        max_velocity_cost = exp(-max_velocity_cost1) + exp(-max_velocity_cost2)

        # movement cost
        movement_cost1 = 0
        for t in range(1, self._time_interval_num+1):

            movement_cost1 += ((q1[:, t] - q1[:, t-1]).T @
                                movement_W1 @
                               (q1[:, t] - q1[:, t-1]))
            '''movement_cost1 += ((q1[:, self._time_interval_num] - robot1_q0).T @
                                movement_W1 @
                                (q1[:, self._time_interval_num] - robot1_q0))'''

        movement_cost2 = 0
        for t in range(1, self._time_interval_num + 1):
            movement_cost2 += ((q2[:, t] - q2[:, t - 1]).T @
                                movement_W2 @
                               (q2[:, t] - q2[:, t - 1]))

            '''movement_cost2 += ((q2[:, self._time_interval_num] - robot2_q0).T @
                                    movement_W2 @
                                    (q2[:, self._time_interval_num] - robot2_q0))'''

        movement_cost = movement_cost1 + movement_cost2

        # input cost
        input_cost1 = 0
        for i in range(self._time_interval_num):
            input_cost1 += (q1_ddot[:, i].T@
                            input_W1 @
                            q1_ddot[:, i])

        input_cost2 = 0
        for i in range(self._time_interval_num):
            input_cost2 += (q2_ddot[:, i].T @
                            input_W2 @
                            q2_ddot[:, i])

        input_cost = input_cost1 + input_cost2

        #time cost
        time_cost = 0
        time_cost += time_W * (time ** 2)

        obj = rel_distance_cost + rel_orientation_cost + max_velocity_cost + movement_cost + time_cost + input_cost
        return obj

    def __define_constraints__(self,
                               robot1:Robot,
                               robot2:Robot,
                               robot1_q0,
                               robot2_q0,
                               rel_pos_target,
                               rel_ori_target):
        w = []
        w0 = []
        lbw = []
        ubw = []
        g = []
        lbg = []
        ubg = []

        t_opt = []
        x1_opt = []
        x1_dot_opt = []
        u1_opt = []
        x2_opt = []
        x2_dot_opt = []
        u2_opt = []

        # unpack variable
        time = self._variable['time']
        q1 = self._variable['q1']
        q1_dot = self._variable['q1_dot']
        q1_ddot = self._variable['q1_ddot']
        q2 = self._variable['q2']
        q2_dot = self._variable['q2_dot']
        q2_ddot = self._variable['q2_ddot']

        # robot state variable constraints
        T_fk1 = robot1.get_position_fk()
        T_fk2 = robot2.get_position_fk()
        q_fk1 = robot1.get_quaternion_fk()
        q_fk2 = robot2.get_quaternion_fk()

        w += [q1[:, 0]]
        lbw += robot1_q0
        ubw += robot1_q0
        w0 += robot1_q0
        x1_opt.append(q1[:, 0])

        w += [q2[:, 0]]
        lbw += robot2_q0
        ubw += robot2_q0
        w0 += robot2_q0
        x2_opt.append(q2[:, 0])

        for t in range(self._time_interval_num):
            if t == self._time_interval_num-1:
                w += [q1[:, self._time_interval_num]]
                lbw += robot1.q_min
                ubw += robot1.q_max
                w0 +=  robot1_q0

                w += [q2[:, self._time_interval_num]]
                lbw += robot2.q_min
                ubw += robot2.q_max
                w0 += robot2_q0

                # make sure get to the target point
                tcp_orientation1 = q_fk1(q1[:, t])
                tcp_orientation2 = q_fk2(q2[:, t])
                rm1 = quaternion2rm(tcp_orientation1)
                rm2 = quaternion2rm(tcp_orientation2)
                rel_rm = calculate_rel_rm(rm2, rm1)
                # rel_rm = calculate_rel_rm(rm1, rm2)
                g += [compare_2_rm(rel_rm, rel_ori_target)]
                lbg += [-0.0001]
                ubg += [0.0001]

                tcp_position1 = T_fk1(q1[:, t])[:3, 3]
                tcp_position2 = T_fk2(q2[:, t])[:3, 3]
                rel_position = mtimes(rm2.T, (tcp_position1 - tcp_position2))
                # rel_position = mtimes(rm1.T, (tcp_position2 - tcp_position1))
                g += [rel_position - rel_pos_target]
                lbg += [-0.0001, -0.0001, -0.0001]
                ubg += [0.0001, 0.0001, 0.0001]

            else:
                w += [q1[:, t + 1]]
                lbw += robot1.q_min
                ubw += robot1.q_max
                w0 += robot1_q0

                w += [q2[:, t + 1]]
                lbw += robot2.q_min
                ubw += robot2.q_max
                w0 += robot2_q0
            x1_opt.append(q1[:, t + 1])
            x2_opt.append(q2[:, t + 1])

        #velocity constraints
        for t in range(self._time_interval_num+1):
            if t == self._time_interval_num or t == 0:
                w += [q1_dot[:, t]]
                lbw += [0] * robot1.get_joint_num()
                ubw += [0] * robot1.get_joint_num()
                w0 += [0] * robot1.get_joint_num()

                w += [q2_dot[:, t]]
                lbw += [0] * robot2.get_joint_num()
                ubw += [0] * robot2.get_joint_num()
                w0 += [0] * robot2.get_joint_num()

            else:
                w += [q1_dot[:, t]]
                lbw += [-lim for lim in robot1.velocity_limits]
                ubw += robot1.velocity_limits
                w0 += [0] * robot1.get_joint_num()

                w += [q2_dot[:, t]]
                lbw += [-lim for lim in robot2.velocity_limits]
                ubw += robot2.velocity_limits
                w0 += [0] * robot2.get_joint_num()
            x1_dot_opt.append(q1_dot[:, t])
            x2_dot_opt.append(q2_dot[:, t])

        # input constraints
        for t in range(self._time_interval_num):
            w += [q1_ddot[:, t]]
            lbw += [-lim for lim in robot1.acc_limits]
            ubw += robot1.acc_limits
            w0 += [0] * robot1.get_joint_num()

            w += [q2_ddot[:, t]]
            lbw += [-lim for lim in robot2.acc_limits]
            ubw += robot2.acc_limits
            w0 += [0] * robot2.get_joint_num()

            u1_opt.append(q1_ddot[:, t])
            u2_opt.append(q2_ddot[:, t])

        # time constraints
        w += [time]
        lbw += [0]
        ubw += [10]
        w0 += [0]
        t_opt.append(time)

        # dynamic constraints
        for t in range(1, self._time_interval_num+1):
            g += [q1[:, t] - q1[:, t - 1] -
                  q1_dot[:, t-1] *  (time / self._time_interval_num) -
                  1 / 2 * q1_ddot[:, t - 1] * ((time / self._time_interval_num) ** 2)]
            lbg += [0] * robot1.get_joint_num()
            ubg += [0] * robot1.get_joint_num()
            g += [q1_dot[:, t] - q1_dot[:, t - 1] - q1_ddot[:, t - 1] * (time / self._time_interval_num)]
            lbg += [0] * robot1.get_joint_num()
            ubg += [0] * robot1.get_joint_num()

            g += [q2[:, t] - q2[:, t - 1] -
                  q2_dot[:, t - 1] * (time / self._time_interval_num) -
                  1 / 2 * q2_ddot[:, t - 1] * ((time / self._time_interval_num) ** 2)]
            lbg += [0] * robot2.get_joint_num()
            ubg += [0] * robot2.get_joint_num()
            g += [q2_dot[:, t] - q2_dot[:, t - 1] - q2_ddot[:, t - 1] * (time / self._time_interval_num)]
            lbg += [0] * robot2.get_joint_num()
            ubg += [0] * robot2.get_joint_num()

        # return
        constraints = {
            'w': w,
            'lbw': lbw,
            'ubw': ubw,
            'w0': w0,
            'g': g,
            'lbg': lbg,
            'ubg': ubg,
            'x1_opt': x1_opt,
            'x1_dot_opt': x1_dot_opt,
            'u1_opt': u1_opt,
            'x2_opt': x2_opt,
            'x2_dot_opt': x2_dot_opt,
            'u2_opt': u2_opt,
            't_opt': t_opt
        }
        return constraints



