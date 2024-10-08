import numpy as np
from casadi import *


class TrajectoryPolynomials:
    def __init__(self,
                 poly_dim:int=3,
                 traj_dim:int=7,
                 poly_type='Lagrange'):
        self._poly_dim = poly_dim
        self._traj_dim = traj_dim
        self._poly_type = poly_type

        self.roots = np.append(0, collocation_points(self._poly_dim, 'legendre'))

        self.collocation_ceo = np.zeros((self._poly_dim + 1, self._poly_dim + 1))
        self.collocation_acc_ceo = np.zeros((self._poly_dim + 1, self._poly_dim + 1))
        self.continuity_ceo = np.zeros(self._poly_dim + 1)
        self.quadrature_ceo = np.zeros(self._poly_dim + 1)
        self.__define_polynomial_basis()

    def get_polynomial_dim(self):
        return self._poly_dim

    def get_trajectory_dim(self):
        return self._traj_dim

    def __define_polynomial_basis(self):
        for j in range(self._poly_dim + 1):
            l = np.poly1d([1])
            for r in range(self._poly_dim + 1):
                if r != j:
                    l *= np.poly1d([1, -self.roots[r]]) / (self.roots[j] - self.roots[r])
            self.continuity_ceo[j] = l(1.0)

            d_l = np.polyder(l)
            for r in range(self._poly_dim + 1):
                self.collocation_ceo[j, r] = d_l(self.roots[r])
            integral_l = np.polyint(l)
            self.quadrature_ceo[j] = integral_l(1.0)

            dd_l = np.polyder(d_l)
            for r in range(self._poly_dim + 1):
                self.collocation_acc_ceo[j, r] = dd_l(self.roots[r])


