import numpy as np
from itertools import product
from qutip import *


class SystemHamiltonian:



    def __init__(self, E, J, K, dipole_single, dipole_x, lam_single,
                 coupling_sites,
                 rate_rad=None, rate_norad=None):

        # ---------- system parameters ----------
        self.hbar = 5308.8    # cm^-1 * fs
        self.E = E
        self.J = J
        self.K = K
        self.N = J.shape[0]

        self.dipole_single = dipole_single
        self.dipole_x = dipole_x
        self.lam_single = lam_single
        self.coupling_sites = coupling_sites

        # ---------- rates ----------
        if rate_rad is None:
            rate_rad = self.hbar * np.array([0, 1/10e6, 1/1e6])

        if rate_norad is None:
            rate_norad = self.hbar *np.array([0, 1/4.2e6, 1/100])

        self.rate_rad = self.hbar *rate_rad
        self.rate_norad = self.hbar* rate_norad

        self.alpha = np.divide(
            rate_rad,
            rate_norad,
            out=np.zeros_like(rate_rad),
            where=rate_norad != 0
        )

        self.phi = np.divide(
            rate_rad,
            rate_rad + rate_norad,
            out=np.zeros_like(rate_rad),
            where=(rate_rad + rate_norad) != 0
        )

        # ---------- build operators ----------
        self.ham_sys, self.labels = self.hamiltonian_custom_order()
        self.dipole, _ = self.mu_operator_ordered()
        self.ham_sysbath, _ = self.sys_bath_list()

        self.nsite = self.ham_sys.shape[0]

        self.L1, self.L2 = self.Non_adiabatic_L()



    # -------------------------------------------------
    # BASIS REORDERING
    # -------------------------------------------------

    def reorder_and_truncate(self, H, cutoff=2):

        dim = 3
        basis_labels = list(product(range(dim), repeat=self.N))

        filtered = [(i, label, sum(label))
                    for i, label in enumerate(basis_labels)
                    if sum(label) <= cutoff]

        filtered.sort(key=lambda x: (x[2], x[1]))

        indices = [i for i, _, _ in filtered]
        new_labels = [label for _, label, _ in filtered]

        H_trunc = H.full()[np.ix_(indices, indices)]

        return H_trunc, new_labels

    # -------------------------------------------------
    # HAMILTONIAN
    # -------------------------------------------------

    def hamiltonian_custom_order(self):

        g = basis(3, 0)
        e = basis(3, 1)
        f = basis(3, 2)

        H = 0

        # on-site energies
        for i in range(self.N):

            operators = [qeye(3) for _ in range(self.N)]

            operators[i] = (
                self.E[i, 0]*g*g.dag() +
                self.E[i, 1]*e*e.dag() +
                self.E[i, 2]*f*f.dag()
            )

            H += tensor(*operators)

        # J couplings
        for i in range(self.N):
            for j in range(self.N):

                if i != j:

                    op_i = [qeye(3) for _ in range(self.N)]
                    op_j = [qeye(3) for _ in range(self.N)]

                    op_i[i] = e*g.dag()
                    op_j[j] = g*e.dag()

                    H += self.J[i][j] * tensor(*op_i) * tensor(*op_j)

        # K couplings
        for i in range(self.N):
            for j in range(self.N):

                if i != j:

                    op_i = [qeye(3) for _ in range(self.N)]
                    op_j = [qeye(3) for _ in range(self.N)]

                    op_i[i] = e*g.dag()
                    op_j[j] = e*f.dag()

                    aux = tensor(*op_i) * tensor(*op_j)

                    H += self.K[i][j]*aux + self.K[i][j]*aux.dag()

        return self.reorder_and_truncate(H)

    # -------------------------------------------------
    # DIPOLE OPERATOR
    # -------------------------------------------------

    def mu_operator_ordered(self):

        g = basis(3, 0)
        e = basis(3, 1)
        f = basis(3, 2)

        op = (self.dipole_single[0]*(g*e.dag() + e*g.dag()) +
              self.dipole_single[1]*(e*f.dag() + f*e.dag()))

        mu = 0

        for i in range(self.N):

            operators = [qeye(3) for _ in range(self.N)]
            operators[i] = self.dipole_x[i] * op

            mu += tensor(*operators)

        return self.reorder_and_truncate(mu)

    # -------------------------------------------------
    # SYSTEM BATH
    # -------------------------------------------------

    def sys_bath_ordered(self, site):

        g = basis(3, 0)
        e = basis(3, 1)
        f = basis(3, 2)

        op = self.lam_single[0]*e*e.dag() + self.lam_single[1]*f*f.dag()

        operators = [qeye(3) for _ in range(self.N)]
        operators[site] = op

        Q = tensor(*operators)

        return self.reorder_and_truncate(Q)

    def sys_bath_list(self):

        list_Q = []

        for site in self.coupling_sites:
            list_Q.append(self.sys_bath_ordered(site-1)[0])

        labels = self.sys_bath_ordered(0)[1]

        return list_Q, labels

    # -------------------------------------------------
    # NON ADIABATIC OPERATORS
    # -------------------------------------------------

    def Non_adiabatic_L(self):

        g = basis(3, 0)
        e = basis(3, 1)
        f = basis(3, 2)

        L1 = []
        L2 = []

        for i in range(self.N):

            operators = [qeye(3) for _ in range(self.N)]
            operators[i] = g*e.dag()

            aux, _ = self.reorder_and_truncate(tensor(*operators))
            L1.append(aux)

        for i in range(self.N):

            operators = [qeye(3) for _ in range(self.N)]
            operators[i] = e*f.dag()

            aux, _ = self.reorder_and_truncate(tensor(*operators))
            L2.append(aux)

        return L1, L2

    # -------------------------------------------------
    # PROJECTORS
    # -------------------------------------------------

    def Projector(self, manifold):

        g = basis(3, 0)
        e = basis(3, 1)
        f = basis(3, 2)

        P = 0

        for i in range(self.N):

            operators = [qeye(3) for _ in range(self.N)]

            if manifold == 0:
                operators[i] = g*g.dag()

            elif manifold == 1:
                operators[i] = e*e.dag()

            elif manifold == 2:
                operators[i] = f*f.dag()

            P += tensor(*operators)

        P_reordered, labels = self.reorder_and_truncate(P)

        return self.rate_rad[manifold] * P_reordered

    # convenience methods

    def get_P0(self):
        return self.Projector(0)

    def get_P1(self):
        return self.Projector(1)

    def get_P2(self):
        return self.Projector(2)

    # -------------------------------------------------
    # ALPHA MATRIX
    # -------------------------------------------------

    def get_alpha_matrix(self):

        aux_alpha, _ = self.mu_operator_ordered()

        return np.tril(aux_alpha, 0) @ np.triu(aux_alpha, 0)
