from urchin import URDF
from data import H5Reader, H5Writer

data_ref = H5Reader(h5file_path='../data/h5/curobo_ik_solution_relative.h5')
data_opt = H5Reader(h5file_path='../experiments/nlp_optimized_joint.h5', is_reference=False)
robot = URDF.load("../model_file/demo_robdekon_scanning.urdf")
for i in range(data_ref.length):
    robot.show(cfg={
        'shoulder_pan_joint': data_ref.qe_ur[i][0],
        'shoulder_lift_joint': data_ref.qe_ur[i][1],
        'elbow_joint': data_ref.qe_ur[i][2],
        'wrist_1_joint': data_ref.qe_ur[i][3],
        'wrist_2_joint': data_ref.qe_ur[i][4],
        'wrist_3_joint': data_ref.qe_ur[i][5],
        'iiwa_joint_1_right': data_ref.qe_iiwa[i][0],
        'iiwa_joint_2_right': data_ref.qe_iiwa[i][1],
        'iiwa_joint_3_right': data_ref.qe_iiwa[i][2],
        'iiwa_joint_4_right': data_ref.qe_iiwa[i][3],
        'iiwa_joint_5_right': data_ref.qe_iiwa[i][4],
        'iiwa_joint_6_right': data_ref.qe_iiwa[i][5],
        'iiwa_joint_7_right': data_ref.qe_iiwa[i][6],
    })

    robot.show(cfg={
        'shoulder_pan_joint': data_opt.nlp_sol['nlp_qe'][i, 7],
        'shoulder_lift_joint': data_opt.nlp_sol['nlp_qe'][i, 8],
        'elbow_joint': data_opt.nlp_sol['nlp_qe'][i, 9],
        'wrist_1_joint': data_opt.nlp_sol['nlp_qe'][i, 10],
        'wrist_2_joint': data_opt.nlp_sol['nlp_qe'][i, 11],
        'wrist_3_joint': data_opt.nlp_sol['nlp_qe'][i, 12],
        'iiwa_joint_1_right': data_opt.nlp_sol['nlp_qe'][i, 0],
        'iiwa_joint_2_right': data_opt.nlp_sol['nlp_qe'][i, 1],
        'iiwa_joint_3_right': data_opt.nlp_sol['nlp_qe'][i, 2],
        'iiwa_joint_4_right': data_opt.nlp_sol['nlp_qe'][i, 3],
        'iiwa_joint_5_right': data_opt.nlp_sol['nlp_qe'][i, 4],
        'iiwa_joint_6_right': data_opt.nlp_sol['nlp_qe'][i, 5],
        'iiwa_joint_7_right': data_opt.nlp_sol['nlp_qe'][i, 6],
    })
