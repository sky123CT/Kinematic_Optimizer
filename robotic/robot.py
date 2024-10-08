from casadi import *
from urdf2casadi import urdfparser as u2c
franka = ddq_lims = [15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0]

class Robot:
    def __init__(self,
                 robot_joint_num=7,
                 robot_file='dual_franka_panda.urdf',
                 root_link='base_link',
                 end_link='right_ee_link',
                 robot_dir=os.path.abspath(os.path.join(os.getcwd(), "../model_file/")) + '/'):
        # robot doc
        self._robot_file = robot_file
        self._robot_dir = robot_dir

        # robot setting
        self._robot_joint_num = robot_joint_num
        self._root_link = root_link
        self._end_link = end_link
        self._robot_parser = self.__define_robot_parser()
        _, self.velocity_limits = self.__get_dynamics_limits()
        self.acc_limits = ddq_lims
        self._joint_list, _, self.q_max, self.q_min = self.__get_robot_feature()

        # kinematics
        self._position_fk = self.__define_position_fk__()
        self._quaternion_fk = self.__define_quaternion_fk__()

    def get_joint_num(self):
        return self._robot_joint_num

    def get_position_fk(self):
        return self._position_fk

    def get_quaternion_fk(self):
        return self._quaternion_fk

    def generate_random_joint_angles(self):
        random_joint_angles = []
        for i in range(self.get_joint_num()):
            random_joint_angles.append(np.random.uniform(self.q_min[i], self.q_max[i]))
        return random_joint_angles

    def __define_robot_parser(self):
        robot_parser = u2c.URDFparser()
        robot_parser.from_file(self._robot_dir + self._robot_file)
        return robot_parser

    def __get_robot_feature(self):
        joint_list, names, q_max, q_min = self._robot_parser.get_joint_info(self._root_link, self._end_link)
        return joint_list, names, q_max, q_min

    def __get_dynamics_limits(self):
        effort_limit, velocity_limit = self._robot_parser.get_dynamics_limits(self._root_link, self._end_link)
        return effort_limit, velocity_limit

    def __define_position_fk__(self, fk_option='T_fk'):
        fk_dict = self._robot_parser.get_forward_kinematics(self._root_link, self._end_link)
        fk = fk_dict[fk_option]
        return fk

    def __define_quaternion_fk__(self, fk_option='quaternion_fk'):
        fk_dict = self._robot_parser.get_forward_kinematics(self._root_link, self._end_link)
        fk = fk_dict[fk_option]
        return fk


class MultiRobot:
    def __init__(self,
                 arm_num=2,
                 robot_joint_num_list=None,
                 robot_file='dual_franka_panda.urdf',
                 root_link='base_link',
                 end_link_list=None,
                 robot_dir=os.path.abspath(os.path.join(os.getcwd(), "../model_file/")) + '/'):

        if end_link_list is None:
            end_link_list = ['right_ee_link', 'left_ee_link']
        if robot_joint_num_list is None:
            robot_joint_num_list = [7, 7]
        for i in range(arm_num):
            self.__dict__['robot_arm_' + str(i+1)] = Robot(robot_joint_num=robot_joint_num_list[i],
                                                           robot_file=robot_file,
                                                           root_link=root_link,
                                                           end_link=end_link_list[i],
                                                           robot_dir=robot_dir)

if __name__ == '__main__':
    robot = Robot()

