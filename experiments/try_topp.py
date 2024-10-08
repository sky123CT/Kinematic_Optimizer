import toppra as ta
import toppra.constraint as constraint
import toppra.algorithm as algo
import numpy as np
import matplotlib.pyplot as plt
import time

ta.setup_logging("INFO")

################################################################################
# We generate a path with some random waypoints.

def generate_new_problem():
    # Parameters
    way_pts = np.array([[1.1395972967147827, -1.0733050107955933, -2.3388731479644775, -2.0545494556427, -0.05248694121837616, -0.9439312815666199, 0.9866145849227905, -2.577087879180908, 0.7886301875114441, -0.1658760905265808, -1.5295802354812622, -2.0464019775390625, 2.3265230655670166],
                        [2.5537638069616744, -2.0940405174298626, -1.4903735089921177, -1.671064166872525, 0.5131758350312333, -0.7018457356711542, 1.5522809583525268, -2.6678402723056234, 1.7673163172083333, 0.004142297002359613, -1.7049826906000676, -1.480735257174569, 2.4160840899650453]])
    return (
        np.linspace(0, 1, 2),
        way_pts,
        np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 3.15, 3.15, 3.15, 3.2, 3.2, 3.2]),
        np.array([5.0, 5.0, 3.0, 2.0, 2.0, 2.0, 2.0, 5.0, 5.0, 3.0, 2.0, 2.0, 2.0]),
    )
ss, way_pts, v_lims, a_lims = generate_new_problem()

path = ta.SplineInterpolator(ss, way_pts)
pc_vel = constraint.JointVelocityConstraint(v_lims)
pc_acc = constraint.JointAccelerationConstraint(a_lims)

instance = algo.TOPPRA([pc_vel, pc_acc], path, parametrizer="ParametrizeConstAccel")
jnt_traj = instance.compute_trajectory()

ts_sample = np.linspace(0, jnt_traj.duration, 100)
qs_sample = jnt_traj(ts_sample)
qds_sample = jnt_traj(ts_sample, 1)
qdds_sample = jnt_traj(ts_sample, 2)
fig, axs = plt.subplots(3, 1, sharex=True)
for i in range(path.dof):
    # plot the i-th joint trajectory
    axs[0].plot(ts_sample, qs_sample[:, i], c="C{:d}".format(i))
    axs[1].plot(ts_sample, qds_sample[:, i], c="C{:d}".format(i))
    axs[2].plot(ts_sample, qdds_sample[:, i], c="C{:d}".format(i))
axs[2].set_xlabel("Time (s)")
axs[0].set_ylabel("Position (rad)")
axs[1].set_ylabel("Velocity (rad/s)")
axs[2].set_ylabel("Acceleration (rad/s2)")
plt.show()


################################################################################
# Optionally, we can inspect the output.
instance.compute_feasible_sets()
instance.inspect()