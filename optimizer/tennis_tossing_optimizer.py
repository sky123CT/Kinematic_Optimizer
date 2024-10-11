from casadi import *
from .utility import *
import numpy as np
from robotic import *
import copy
import time


class TennisTossingOpt:
    def __init__(self,
                 robot:Robot,
                 release_pos,
                 release_ori,
                 release_vel,
                 robot_q0,
                 collocation_num=3,
                 interpolation_dim_position=3,
                 interpolation_dim_rotation=4,
                 time_interval_num=10,
                 vel_max_W=np.diag([1]*7),
                 movement_W=np.diag([1]*7),
                 input_W=np.diag([1]*7),
                 time_W=np.diag([1]*10)):

        self._collocation_num = collocation_num
        self._collocation_roots = np.append(0, collocation_points(self._collocation_num, 'legendre'))
        self._interpolation_dim_position = interpolation_dim_position
        self._interpolation_dim_rotation = interpolation_dim_rotation

        self._time_interval_num = time_interval_num

        #define variable
        self._variable= self.__define_variables__(
            robot=robot,
            robot_joint_dim=robot.get_joint_num())

        #define obj function
        self.objective_function = self.__define_obj_function__(
            robot=robot,
            vel_max_W=vel_max_W,
            movement_W=movement_W,
            input_W=input_W,
            time_W=time_W)

        #define constraints
        self.constraints = self.__define_constraints__(
            robot=copy.deepcopy(robot),
            robot_q0=robot_q0,
            release_pos=release_pos,
            release_ori=release_ori,
            release_vel=release_vel)

    def optimize(self):
        L = self.objective_function
        w = self.constraints['w']
        lbw = self.constraints['lbw']
        ubw = self.constraints['ubw']
        w0 = self.constraints['w0']
        g = self.constraints['g']
        lbg = self.constraints['lbg']
        ubg = self.constraints['ubg']
        x_opt = self.constraints['x_opt']
        x_dot_opt = self.constraints['x_dot_opt']
        u_opt = self.constraints['u_opt']
        t_opt = self.constraints['t_opt']

        w = vertcat(*w)
        g = vertcat(*g)
        x_opt = horzcat(*x_opt)
        x_dot_opt = horzcat(*x_dot_opt)
        u_opt = horzcat(*u_opt)
        t_opt = horzcat(*t_opt)

        trajectories = Function('trajectories',
                                [w],
                                [x_opt, x_dot_opt, u_opt, t_opt],
                                ['w'],
                                ['x', 'x_dot', 'u', 't'])


        # Create an NLP solver
        prob = {'f': L, 'x': w, 'g': g}
        opti_setting = {
            'ipopt.max_iter': 1000,
            'ipopt.print_level': 3,
            # 'hessian_approximation':'limited-memory',
            'print_time': 3
        }
        solver = nlpsol('solver', 'ipopt', prob, opti_setting)

        # Solve the NLP
        start = time.time()
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        end = time.time()
        optimization_time = end - start
        x_opt, x_dot_opt, u_opt, t_opt = trajectories(sol['x'])

        return x_opt, x_dot_opt, u_opt, t_opt, optimization_time


    def __define_variables__(self,
                             robot:Robot,
                             robot_joint_dim):
        # time
        time = MX.sym('T')

        # robot link theta as state
        q = []
        for i in range(self._time_interval_num + 1):
            qi = []
            for j in range(robot_joint_dim):
                qi.append(MX.sym('q_' + str(j) + '_at_' + str(i)))
            q.append(vertcat(*qi))
        q = horzcat(*q)


        # velocity
        q_dot = []
        for i in range(self._time_interval_num + 1):
            q_dot_i = []
            for j in range(robot_joint_dim):
                q_dot_i.append(MX.sym('q_dot_' + str(j) + '_at_' + str(i)))
            q_dot.append(vertcat(*q_dot_i))
        q_dot = horzcat(*q_dot)

        # control input variable
        q_ddot = []
        for i in range(self._time_interval_num):
            q_ddot_i = []
            for j in range(robot_joint_dim):
                q_ddot_i.append(MX.sym('q_ddot_' + str(j) + '_at_' + str(i)))
            q_ddot.append(vertcat(*q_ddot_i))
        q_ddot = horzcat(*q_ddot)

        robot.get_pos_jac_mapping(q[:, 0])

        variable = {'time': time,
                    'q': q,
                    'q_dot': q_dot,
                    'q_ddot': q_ddot}
        return variable


    def __define_obj_function__(self,
                                robot:Robot,
                                vel_max_W,
                                movement_W,
                                input_W,
                                time_W):
        # unpack variable
        time = self._variable['time']
        q = self._variable['q']
        q_dot = self._variable['q_dot']
        q_ddot = self._variable['q_ddot']

        '''vel_max_cost = 0
        for t in range(1, self._time_interval_num):
            vel_max_cost += ((q_dot[:, t] - robot.velocity_limits).T @
                             vel_max_W @
                             (q_dot[:, t] - robot.velocity_limits))'''

        # movement cost
        movement_cost = 0
        for t in range(1, self._time_interval_num + 1):
            movement_cost += ((q[:, t] - q[:, t-1]).T @
                              movement_W @
                              (q[:, t] - q[:, t-1]))

        # input cost
        input_cost = 0
        for i in range(self._time_interval_num):
            input_cost += (q_ddot[:, i].T @
                            input_W @
                            q_ddot[:, i])

        #time cost
        time_cost = 0
        time_cost += time_W * (time ** 2)

        obj = movement_cost + time_cost + input_cost #  + vel_max_cost
        return obj

    def __define_constraints__(self,
                               robot:Robot,
                               robot_q0,
                               release_pos,
                               release_ori,
                               release_vel):
        w = []
        w0 = []
        lbw = []
        ubw = []
        g = []
        lbg = []
        ubg = []

        t_opt = []
        x_opt = []
        x_dot_opt = []
        u_opt = []

        # unpack variable
        time = self._variable['time']
        q = self._variable['q']
        q_dot = self._variable['q_dot']
        q_ddot = self._variable['q_ddot']

        # robot state variable constraints
        T_fk = robot.get_position_fk()
        q_fk = robot.get_quaternion_fk()

        w += [q[:, 0]]
        lbw += robot.q_min
        ubw += robot.q_max
        w0 += robot_q0
        x_opt.append(q[:, 0])

        for t in range(self._time_interval_num):
            if t == self._time_interval_num / 2:
                w += [q[:, t+1]]
                lbw += robot.q_min
                ubw += robot.q_max
                w0 +=  robot_q0

                # make sure get to the target point
                '''tcp_orientation = q_fk(q[:, t])
                g += [compare_2_quaternion(tcp_orientation, release_ori)]
                lbg += [-0.0001]
                ubg += [0.0001]'''

                tcp_position = T_fk(q[:, t])[:3, 3]
                g += [tcp_position- release_pos]
                lbg += [-0.0001, -0.0001, -0.0001]
                ubg += [0.0001, 0.0001, 0.0001]

            else:
                w += [q[:, t + 1]]
                lbw += robot.q_min
                ubw += robot.q_max
                w0 += robot_q0
            x_opt.append(q[:, t + 1])


        #velocity constraints
        for t in range(self._time_interval_num + 1):
            if t == self._time_interval_num or t == 0:
                w += [q_dot[:, t]]
                lbw += [0] * robot.get_joint_num()
                ubw += [0] * robot.get_joint_num()
                w0 += [0] * robot.get_joint_num()

            elif t == self._time_interval_num/2:
                w += [q_dot[:, t]]
                lbw += [-lim for lim in robot.velocity_limits]
                ubw += robot.velocity_limits
                w0 += robot.velocity_limits

                vel = jtimes(T_fk(q[:, t])[:3, 3], q[:, t], q_dot[:, t])
                g += [vel - release_vel]
                lbg += [-0.0001, -0.0001, -0.0001]
                ubg += [0.0001, 0.0001, 0.0001]

            else:
                w += [q_dot[:, t]]
                lbw += [-lim for lim in robot.velocity_limits]
                ubw += robot.velocity_limits
                w0 += robot.velocity_limits
            x_dot_opt.append(q_dot[:, t])

        # input constraints
        for t in range(self._time_interval_num):
            w += [q_ddot[:, t]]
            lbw += [-lim for lim in robot.acc_limits]
            ubw += robot.acc_limits
            w0 += [0] * robot.get_joint_num()

            u_opt.append(q_ddot[:, t])


        # time constraints
        w += [time]
        lbw += [0]
        ubw += [100]
        w0 += [0]
        t_opt.append(time)

        # dynamic constraints
        for t in range(1, self._time_interval_num+1):
            g += [q[:, t] - q[:, t - 1] -
                  q_dot[:, t-1] *  (time / self._time_interval_num) -
                  1 / 2 * q_ddot[:, t - 1] * ((time / self._time_interval_num) ** 2)]
            lbg += [0] * robot.get_joint_num()
            ubg += [0] * robot.get_joint_num()
            g += [q_dot[:, t] - q_dot[:, t - 1] - q_ddot[:, t - 1] * (time / self._time_interval_num)]
            lbg += [0] * robot.get_joint_num()
            ubg += [0] * robot.get_joint_num()

        # return
        constraints = {
            'w': w,
            'lbw': lbw,
            'ubw': ubw,
            'w0': w0,
            'g': g,
            'lbg': lbg,
            'ubg': ubg,
            'x_opt': x_opt,
            'x_dot_opt': x_dot_opt,
            'u_opt': u_opt,
            't_opt': t_opt
        }
        return constraints
