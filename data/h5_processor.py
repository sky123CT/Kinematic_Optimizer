import h5py
import casadi


class H5Reader:
    def __init__(self,
                 h5file_path='.h5/curobo_ik_solution.h5',
                 is_reference=True):
        if is_reference:
            self.q0_iiwa = []
            self.q0_ur = []
            self.pos_e_iiwa = []
            self.pos_e_ur = []
            self.ori_e_iiwa = []
            self.ori_e_ur = []
            self.qe_iiwa = []
            self.qe_ur = []

            with h5py.File(h5file_path, 'r') as f:
                for group_name in f.keys():
                    grp = f[group_name]
                    self.q0_ur.append(grp['starting_joint_angle'][:6].squeeze().tolist())
                    self.q0_iiwa.append(grp['starting_joint_angle'][6:].squeeze().tolist())
                    self.pos_e_iiwa.append(casadi.DM(grp['goal_tool_position'][:]).T)
                    self.pos_e_ur.append(casadi.DM(grp['goal_position'][:]).T)
                    self.ori_e_iiwa.append(casadi.DM(grp['goal_tool_quaternion'][:]).T)
                    self.ori_e_ur.append(casadi.DM(grp['goal_quaternion'][:]).T)
                    self.qe_ur.append(grp['relative_curobo_joint_angle'][:, :6].squeeze().tolist())
                    self.qe_iiwa.append(grp['relative_curobo_joint_angle'][:, 6:].squeeze().tolist())

            self.length = len(self.q0_iiwa)
        else:
            self.nlp_sol = {}

            with h5py.File(h5file_path, 'r') as f:
                for name in f.keys():
                    self.nlp_sol[name] = f[name][:]

class H5Writer:
    def __init__(self,
                 file_name='nlp_optimized_joint.h5'):
        self.file = h5py.File(file_name, mode='w')

    def write_data(self, data:dict):
         names=locals()
         i = 0
         for i, (key, value) in enumerate(data.items()):
             names['data_' + key] = self.file.create_dataset(key, data=value)
         self.file.close()


if __name__ == '__main__':
    data = {'a':1, 'b':2, 'c':3, 'd':4, 'e':5}
    writer = H5Writer()
    writer.write_data(data)
    with h5py.File("./nlp_optimized_joint.h5", 'r') as f:
        for name in f.keys():
            # print(name)
            print(f[name])