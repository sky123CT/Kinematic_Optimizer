from casadi import *
from .utility import *
import numpy as np
from robotic import *
import copy


class DirectCollocationWT:
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
                 movement_W=np.array([1]*7),
                 max_v_W=np.array([1]*7),
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
        self._t, self._q, self._q_dot, self._q_ddot, self._qc, self._qc_d, self._qc_dd, self._qc_end = self.__define_variables(
            trajectory_polynomials=copy.deepcopy(trajectory_polynomials),
        )

        self.objective_function = self.__define_obj_function__(
            robot=copy.deepcopy(robot),
            traj_poly=copy.deepcopy(trajectory_polynomials),
            pos_target=pos_target,
            ori_target=ori_target,
            distance_W=distance_W,
            orientation_W=orientation_W,
            movement_W=movement_W,
            max_v_W=max_v_W,
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
        xc_opt = self.constraints['xc_opt']
        x_dot_opt = self.constraints['x_dot_opt']
        u_opt = self.constraints['u_opt']
        t_opt = self.constraints['t_opt']

        w = vertcat(*w)
        g = vertcat(*g)
        x_opt = horzcat(*x_opt)
        xc_opt = horzcat(*xc_opt)
        x_dot_opt = horzcat(*x_dot_opt)
        u_opt = horzcat(*u_opt)
        t_opt = horzcat(*t_opt)
        trajectories = Function('trajectories', [w], [x_opt, xc_opt, x_dot_opt, u_opt, t_opt], ['w'], ['x', 'xc', 'x_dot', 'u', 't'])


        # Create an NLP solver
        prob = {'f': L, 'x': w, 'g': g}
        opti_setting = {
            'ipopt.max_iter': 1000000,
            'ipopt.print_level': 3,
            'print_time': 3
        }
        solver = nlpsol('solver', 'ipopt', prob, opti_setting)

        # Solve the NLP
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        x_opt, xc_opt, x_dot_opt, u_opt, t_opt = trajectories(sol['x'])
        #print(x_opt.full())
        #print(u_opt.full())
        #print(t_opt.full())

        return x_opt, xc_opt, x_dot_opt, u_opt, t_opt


    def __define_variables(self,
                           trajectory_polynomials:TrajectoryPolynomials,
                           robot_joint_dim: int = 7):
        # time
        time = MX.sym('T')

        # robot link theta as state
        q = []
        for i in range(self._time_interval_num + 1):
            qi = []
            for j in range(robot_joint_dim):
                qi.append(MX.sym('q' + str(j) + '_at_' + str(i)))
            q.append(vertcat(*qi))
        q = horzcat(*q)

        # velocity
        q_dot = []
        for i in range(self._time_interval_num+1):
            q_dot_i = []
            for j in range(robot_joint_dim):
                q_dot_i.append(MX.sym('q_ddot_' + str(j) + '_at_' + str(i)))
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

        # define collocation variables
        qc = []
        qc_end = []
        qc_d = []
        qc_dd = []
        for t in range(self._time_interval_num):
            qc_ti = []
            for i in range(trajectory_polynomials.get_polynomial_dim()):
                qc_tij = []
                for j in range(trajectory_polynomials.get_trajectory_dim()):
                    qc_tij.append(MX.sym('qc_' + str(t) + '_' + str(i) + '_' + str(j)))
                qc_ti.append(vertcat(*qc_tij))
            qc.append(horzcat(*qc_ti))

            q_end = trajectory_polynomials.continuity_ceo[0] * q[:, t]
            qc_d_points = []
            for i in range(1, trajectory_polynomials.get_polynomial_dim()+1):
                d_qc = trajectory_polynomials.collocation_ceo[0, i] * q[:, t]
                for j in range(trajectory_polynomials.get_polynomial_dim()):
                    d_qc = d_qc + trajectory_polynomials.collocation_ceo[j+1, i] * qc[t][:, j]
                qc_d_points.append(d_qc)
                q_end = q_end + trajectory_polynomials.continuity_ceo[i] * qc[t][:, i-1]
            qc_d.append(horzcat(*qc_d_points))
            qc_end.append(q_end)

            qc_dd_points = []
            for i in range(1, trajectory_polynomials.get_polynomial_dim() + 1):
                dd_qc = trajectory_polynomials.collocation_acc_ceo[0, i] * q[:, t]
                for j in range(trajectory_polynomials.get_polynomial_dim()):
                    dd_qc = dd_qc + trajectory_polynomials.collocation_acc_ceo[j + 1, i] * qc[t][:, j]
                qc_dd_points.append(dd_qc)
            qc_dd.append(horzcat(*qc_dd_points))

        return time, q, q_dot, q_ddot, qc, qc_d, qc_dd, qc_end


    def __define_obj_function__(self,
                                robot:Robot,
                                traj_poly:TrajectoryPolynomials,
                                pos_target,
                                ori_target,
                                distance_W,
                                orientation_W,
                                movement_W,
                                max_v_W,
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
                orientation_cost += ((tcp_orientation - ori_target).T @
                                     orientation_W @
                                     (tcp_orientation - ori_target))
                '''orientation_cost += (quaternion_displacement(q1=tcp_orientation, q2=ori_target).T @
                                     orientation_W @
                                     quaternion_displacement(q1=tcp_orientation, q2=ori_target))'''

        #velocity maximizing cost
        max_velocity_cost = 0
        for t in range(1, self._time_interval_num):
            max_velocity_cost += ((self._q_dot[:, t]).T @
                                   max_v_W @
                                  (self._q_dot[:, t]))

        # movement cost
        movement_cost = 0
        for t in range(1, self._time_interval_num+1):
            movement_cost += ((self._q[:, t] - self._q[:, t-1]).T @
                              movement_W @
                              (self._q[:, t] - self._q[:, t-1]))

        # input cost
        input_cost = 0
        for i in range(self._time_interval_num):
            input_cost += (self._q_ddot[:, i].T@
                           input_W @
                           self._q_ddot[:, i])

        #time cost
        time_cost = 0
        time_cost += time_W * (self._t ** 2)

        obj = distance_cost + orientation_cost + exp(-max_velocity_cost) + movement_cost + time_cost
        # (distance_cost + orientation_cost + input_cost + time_cost)
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
        xc_opt = []
        x_dot_opt = []
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
                xc_opt.append(self._qc[t][:, i])
            if t == self._time_interval_num-1:
                w += [self._q[:, self._time_interval_num]]
                lbw += robot.q_min
                ubw += robot.q_max
                w0 +=  self._q0
                # make sure get to the target point
                end_pos = robot.get_position_fk()(self._q[:, self._time_interval_num])[:3, 3]
                end_ori = robot.get_quaternion_fk()(self._q[:, self._time_interval_num])
                g += [end_pos - self._pos_target]
                lbg += [-0.00001, -0.00001, -0.00001]
                ubg += [0.00001, 0.00001, 0.00001]
                g += [end_ori - self._ori_target] # [quaternion_displacement(q1=end_ori, q2=self._ori_target)]
                lbg += [-0.00001, -0.00001, -0.00001, -0.00001]
                ubg += [0.00001, 0.00001, 0.00001, 0.00001]
            else:
                w += [self._q[:, t + 1]]
                lbw += robot.q_min
                ubw += robot.q_max
                w0 += self._q0
            x_opt.append(self._q[: ,t + 1])

        #velocity constraints
        for t in range(self._time_interval_num+1):
            if t == self._time_interval_num or t == 0:
                w += [self._q_dot[:, t]]
                lbw += [0] * robot.get_joint_num()
                ubw += [0] * robot.get_joint_num()
                w0 += [0] * robot.get_joint_num()
            else:
                w += [self._q_dot[:, t]]
                lbw += [-lim for lim in robot.velocity_limits]
                ubw += robot.velocity_limits
                w0 += [0] * robot.get_joint_num()
            x_dot_opt.append(self._q_dot[:, t])

        # input constraints
        for t in range(self._time_interval_num):
            w += [self._q_ddot[:, t]]
            lbw += [-lim for lim in robot.acc_limits]
            ubw += robot.acc_limits
            w0 += [0] * robot.get_joint_num() #robot.acc_limits
            u_opt.append(self._q_ddot[:, t])

        # time constraints
        w += [self._t]
        lbw += [0]
        ubw += [2]
        w0 += [0]
        t_opt.append(self._t)


        # dynamic constraints
        for t in range(1, self._time_interval_num+1):
            g += [self._q[:, t] - self._q[:, t - 1] -
                  self._q_dot[:, t-1] *  (self._t / self._time_interval_num) -
                  1 / 2 * self._q_ddot[:, t - 1] * ((self._t / self._time_interval_num) ** 2)]
            # g += [self._q[:, t] - self._q[:, t-1] - 1/2 * self._q_ddot[:, t-1] * ((self._t / self._time_interval_num) ** 2)]
            lbg += [0] * robot.get_joint_num()
            ubg += [0] * robot.get_joint_num()

            g += [self._q_dot[:, t] - self._q_dot[:, t - 1] - self._q_ddot[:, t - 1] * (self._t / self._time_interval_num)]
            # g += [self._q[:, t] - self._q[:, t-1] - 1/2 * self._q_ddot[:, t-1] * ((self._t / self._time_interval_num) ** 2)]
            lbg += [0] * robot.get_joint_num()
            ubg += [0] * robot.get_joint_num()

        '''# polynomials constraints
        for t in range(self._time_interval_num):
            for i in range(traj_poly.get_polynomial_dim()):
                g += [self._q_dot[:, t] * (self._t/self._time_interval_num) - self._qc_d[t][:, i]]
                lbg += [0] * robot.get_joint_num()
                ubg += [0] * robot.get_joint_num()

                g += [self._q_ddot[:, t] * (self._t / self._time_interval_num)  - self._qc_dd[t][:, i]]
                lbg += [0] * robot.get_joint_num()
                ubg += [0] * robot.get_joint_num()

            g += [self._qc_end[t] - self._q[:, t + 1]]
            lbg += [0] * robot.get_joint_num()
            ubg += [0] * robot.get_joint_num()'''
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
            'xc_opt': xc_opt,
            'x_dot_opt': x_dot_opt,
            'u_opt': u_opt,
            't_opt': t_opt
        }
        return constraints



