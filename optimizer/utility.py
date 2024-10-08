from liecasadi import *
from casadi import *
import numpy as np
import math


def quaternion_displacement(q1, q2):
    #q1 = MX(q1)
    #q2 = MX(q2)
    inv_q1= quaternion_inverse(q1)
    m = Quaternion.product(inv_q1, q2)
    return quaternion_log(m)


def quaternion_log(q):
    v = np.array(q[1:]).squeeze()
    if abs(q[0] < 1.0):
        a = 1
        try:
            a = math.acos(q[0])
        except:
            pass
        sina = math.sin(a)
        if abs(sina >= 0.005):
            c = a / sina
            v[0] *= c
            v[1] *= c
            v[2] *= c
    return v.tolist()

def quaternion_inverse(q):
    return quaternion_conjugate(q) / Quaternion.product(q, q)

def quaternion_conjugate(q):
    return vertcat(q[0], -q[1], -q[2], -q[3])

def calculate_rel_rm(rm1, rm2):
    return rm1.T @ rm2

def quaternion2rm(q):
    if isinstance(q, DM):
        rm = casadi.DM(3, 3)
    else:
        rm = casadi.MX( 3, 3)
    rm[0, 0] = 1 - 2 * (q[2] ** 2) - 2 * (q[3] ** 2)
    rm[0, 1] = 2 * q[1] * q[2] - 2 * q[0] * q[3]
    rm[0, 2] = 2 * q[1] * q[3] + 2 * q[0] * q[2]
    rm[1, 0] = 2 * q[1] * q[2] + 2 * q[0] * q[3]
    rm[1, 1] = 1 - 2 * (q[1] ** 2) - 2 * (q[3] ** 2)
    rm[1, 2] = 2 * q[2] * q[3] - 2 * q[0] * q[1]
    rm[2, 0] = 2 * q[1] * q[3] - 2 * q[0] * q[2]
    rm[2, 1] = 2 * q[2] * q[3] + 2 * q[0] * q[1]
    rm[2, 2] = 1 - 2 * (q[1] ** 2) - 2 * (q[2] ** 2)
    return rm

def compare_2_rm(rm1, rm2):
    if not isinstance(rm1, MX) and not isinstance(rm2, MX) and not isinstance(rm1, DM) and not isinstance(rm2, DM):
        raise "wrong input type!"
    return casadi.acos((casadi.trace(casadi.mtimes(rm1.T, rm2)) - 1) / 2)

def compare_2_quaternion(q1, q2):
    rm1 = quaternion2rm(q=q1)
    rm2 = quaternion2rm(q=q2)
    return compare_2_rm(rm1, rm2)

