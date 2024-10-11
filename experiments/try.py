from casadi import *
import numpy as np

x = MX.sym("x", 2)
f = x[0] + x[1] +x[0] * x[1]
jac = jacobian(f, x)
print(jac)