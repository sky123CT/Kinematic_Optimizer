import casadi
from casadi import *
from .utility import *
import numpy as np
from robotic import *
import copy


class DirectCollocation:
    def __init__(self,
                 robot:Robot,
                 trajectory_polynomials:TrajectoryPolynomials,
                 pos_target,
                 ori_target,
                 q0,
                 collocation_num=3,
                 interpolation_dim_position=3,
                 interpolation_dim_rotation=4,
                 time_interval_num=10,
                 distance_W=np.array([1, 1, 1]),
                 orientation_W=np.array([1, 1, 1, 1]),
                 input_W=np.array([1]*7),
                 time_W=np.array([1]*10)):

        self._collocation_num = collocation_num
        self._collocation_roots = np.append(0, collocation_points(self._collocation_num, 'legendre'))
        self._interpolation_dim_position = interpolation_dim_position
        self._interpolation_dim_rotation = interpolation_dim_rotation

        self._time_interval_num = time_interval_num
        self._pos_target = pos_target
        self._ori_target = ori_target
        self._q0 = q0
        self._t, self._q, self._qc, self._qc_der, self._qc_end,self._q_dot = self.__define_variables(
            trajectory_polynomials=copy.deepcopy(trajectory_polynomials),
        )

        self.objective_function = self.__define_obj_function__(
            robot=copy.deepcopy(robot),
            traj_poly=copy.deepcopy(trajectory_polynomials),
            pos_target=pos_target,
            ori_target=ori_target,
            distance_W=distance_W,
            orientation_W=orientation_W,
            input_W=input_W,
            time_W=time_W)

        self.constraints = self.__define_constraints__(
            robot=copy.deepcopy(robot),
            traj_poly=copy.deepcopy(trajectory_polynomials)
        )

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
        u_opt = self.constraints['u_opt']
        t_opt = self.constraints['t_opt']

        w = vertcat(*w)
        g = vertcat(*g)
        x_opt = horzcat(*x_opt)
        u_opt = horzcat(*u_opt)
        t_opt = horzcat(*t_opt)
        trajectories = Function('trajectories', [w], [x_opt, u_opt, t_opt], ['w'], ['x', 'u', 't'])


        # Create an NLP solver
        prob = {'f': L, 'x': w, 'g': g}
        opti_setting = {
            # 'ipopt.max_iter': 20,
            'ipopt.print_level': 3,
            'print_time': 3
        }
        solver = nlpsol('solver', 'ipopt', prob, opti_setting)

        # Solve the NLP
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        x_opt, u_opt, t_opt = trajectories(sol['x'])
        #print(x_opt.full())
        #print(u_opt.full())
        #print(t_opt.full())

        return x_opt, u_opt, t_opt


    def __define_variables(self,
                           trajectory_polynomials:TrajectoryPolynomials,
                           robot_joint_dim: int = 7):
        # time variable
        time = []
        for t in range(self._time_interval_num):
            time.append(MX.sym('t_' + str(t)))
        time = vertcat(*time)

        # robot link theta as state
        q = []
        for i in range(self._time_interval_num + 1):
            qi = []
            for j in range(robot_joint_dim):
                qi.append(MX.sym('q' + str(j) + '_at_' + str(i)))
            q.append(vertcat(*qi))
        q = horzcat(*q)

        # control input variable
        q_dot = []
        for i in range(self._time_interval_num):
            q_doti = []
            for j in range(robot_joint_dim):
                q_doti.append(MX.sym('q_dot_' + str(j) + '_at_' + str(i)))
            q_dot.append(vertcat(*q_doti))
        q_dot = horzcat(*q_dot)

        # define collocation variables
        qc = []
        qc_end = []
        qc_der = []
        for t in range(self._time_interval_num):
            qc_ti = []
            for i in range(trajectory_polynomials.get_polynomial_dim()):
                qc_tij = []
                for j in range(trajectory_polynomials.get_trajectory_dim()):
                    qc_tij.append(MX.sym('qc_' + str(t) + '_' + str(i) + '_' + str(j)))
                qc_ti.append(vertcat(*qc_tij))
            qc.append(horzcat(*qc_ti))

            q_end = trajectory_polynomials.continuity_ceo[0] * q[:, t]
            qc_der_points = []
            for i in range(1, trajectory_polynomials.get_polynomial_dim()+1):
                dqc = trajectory_polynomials.collocation_ceo[0, i] * q[:, t]
                for j in range(trajectory_polynomials.get_polynomial_dim()):
                    dqc = dqc + trajectory_polynomials.collocation_ceo[j+1, i] * qc[t][:, j]
                qc_der_points.append(dqc)
                q_end = q_end + trajectory_polynomials.continuity_ceo[i] * qc[t][:, i-1]
            qc_der.append(horzcat(*qc_der_points))
            qc_end.append(q_end)

        return time, q, qc, qc_der, qc_end, q_dot


    def __define_obj_function__(self,
                                robot:Robot,
                                traj_poly:TrajectoryPolynomials,
                                pos_target,
                                ori_target,
                                distance_W,
                                orientation_W,
                                input_W,
                                time_W):
        # cartesian distance
        T_fk = robot.get_position_fk()
        distance_cost = 0
        for t in range(self._time_interval_num):
            for i in range(traj_poly.get_polynomial_dim()):
                tcp_position = T_fk(self._qc[t][:, i])
                distance_cost += ((tcp_position[:3, 3] - pos_target).T @
                                  distance_W @
                                  (tcp_position[:3, 3] - pos_target))

        # quaternion orientation
        q_fk = robot.get_quaternion_fk()
        orientation_cost = 0
        for t in range(self._time_interval_num):
            for i in range(traj_poly.get_polynomial_dim()):
                tcp_orientation = q_fk(self._qc[t][:, i])
                orientation_cost += (quaternion_displacement(q1=tcp_orientation, q2=ori_target).T @
                                     orientation_W @
                                     quaternion_displacement(q1=tcp_orientation, q2=ori_target))

        # input cost
        input_cost = 0
        for i in range(self._time_interval_num):
            input_cost += (self._q_dot[:, i].T@
                           input_W @
                           self._q_dot[:, i])

        # time cost
        time_cost = 0
        time_cost += time_W * (sum1(self._t)**2)

        obj = (distance_cost +
               orientation_cost +
               input_cost +
               time_cost)
        return obj

    def __define_constraints__(self,
                               robot:Robot,
                               traj_poly:TrajectoryPolynomials):
        w = []
        w0 = []
        lbw = []
        ubw = []
        g = []
        lbg = []
        ubg = []

        t_opt = []
        x_opt = []
        u_opt = []

        # state (collocation points) variable constraints
        w += [self._q[:, 0]]
        lbw += self._q0
        ubw += self._q0
        w0 += self._q0
        x_opt.append(self._q[:, 0])
        for t in range(self._time_interval_num):
            for i in range(traj_poly.get_polynomial_dim()):
                w += [self._qc[t][:, i]]
                lbw += robot.q_min
                ubw += robot.q_max
                w0 += self._q0
            if t == self._time_interval_num - 1:
                w += [self._q[:, self._time_interval_num]]
                lbw += robot.q_min
                ubw += robot.q_max
                w0 += self._q0
                # make sure get to the target point
                end_pos = robot.get_position_fk()(self._q[:, self._time_interval_num])[:3, 3]
                end_ori = robot.get_quaternion_fk()(self._q[:, self._time_interval_num])
                g += [end_pos - self._pos_target]
                lbg += [-0.0005, -0.0005, -0.0005]
                ubg += [0.0005, 0.0005, 0.0005]
                g += [end_ori - self._ori_target] # [quaternion_displacement(q1=end_ori, q2=self._ori_target)]
                lbg += [-0.00001, -0.00001, -0.00001, -0.00001]
                ubg += [0.00001, 0.00001, 0.00001, 0.00001]

            else:
                w += [self._q[:, t + 1]]
                lbw += robot.q_min
                ubw += robot.q_max
                w0 += self._q0
            x_opt.append(self._q[: ,t + 1])

        # input variable constraints
        for t in range(self._time_interval_num):
            w += [self._q_dot[:, t]]
            lbw += [-lim for lim in robot.velocity_limits]
            ubw += robot.velocity_limits
            w0 += [0] * robot.get_joint_num()
            u_opt.append(self._q_dot[:, t])

        # time constraints
        w += [self._t]
        lbw += [0] * self._time_interval_num
        ubw += [inf] * self._time_interval_num
        w0 += [0] * self._time_interval_num
        t_opt.append(self._t)

        # polynomials constraints
        for t in range(self._time_interval_num):
            for i in range(traj_poly.get_polynomial_dim()):
                g += [self._q_dot[:, t] * self._t[t] - self._qc_der[t][:, i]]
                lbg += [0] * robot.get_joint_num()
                ubg += [0] * robot.get_joint_num()

            g += [self._qc_end[t] - self._q[:, t + 1]]
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
            'u_opt': u_opt,
            't_opt': t_opt
        }
        return constraints



