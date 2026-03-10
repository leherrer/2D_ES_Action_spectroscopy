import numpy as np
from qutip import *


class RedfieldPropagator:

    def __init__(self, system,
                 lam,
                 gamma,
                 kT):

        self.system = system

        # bath parameters
        self.lam = lam
        self.gamma = gamma
        self.kT = kT
        self.b = 1 / kT

        self.rate_norad = system.rate_norad

        # ---------- system operators ----------
        self.Hsys = Qobj(system.ham_sys)
        self.dipole = Qobj(system.dipole)

        self.mu_p = np.tril(system.dipole, 0)
        self.mu_m = np.triu(system.dipole, 0)

        self.nsite = system.nsite
        self.nbath = len(system.ham_sysbath)
        self.nx = system.N

        # bath operators
        self.O_list = [Qobj(Q) for Q in system.ham_sysbath]

        # projectors
        self.P0 = Qobj(system.get_P0())
        self.P1 = Qobj(system.get_P1())
        self.P2 = Qobj(system.get_P2())



        # non-adiabatic operators
        self.L1 = system.L1
        self.L2 = system.L2

        # build Redfield tensor
        self.RD = self.build_redfield_tensor()

    # -------------------------------------------------
    # Spectral density
    # -------------------------------------------------

    def calculate_DL(self, w):

        if w == 0:

            DL = 2 * np.pi * 2.0 * self.lam / (np.pi * self.gamma * self.b)

        else:

            DL = 2 * np.pi * (
                2.0 * self.lam * self.gamma * w /
                (np.pi * (w**2 + self.gamma**2))
            ) * ((1 / (np.exp(w * self.b) - 1)) + 1)

        return DL

    # -------------------------------------------------
    # Build Redfield tensor
    # -------------------------------------------------

    def build_redfield_tensor(self):

        a_ops = [[Q, self.calculate_DL] for Q in self.O_list]

        RD = bloch_redfield_tensor(
            self.Hsys,
            a_ops=a_ops,
            sec_cutoff=-1,
            fock_basis=True
        )

        # ----- Liouvillian -----

        HL = liouvillian(self.Hsys)

        LL = HL * 0

        for i in range(self.nx):

            L1 = Qobj(self.L1[i])
            L2 = Qobj(self.L2[i])

            L1_sup = self.rate_norad[1] * (
                spre(L1)*spost(L1.dag())
                - 0.5*(spre(L1.dag()*L1) + spost(L1.dag()*L1))
            )

            L2_sup = self.rate_norad[2] * (
                spre(L2)*spost(L2.dag())
                - 0.5*(spre(L2.dag()*L2) + spost(L2.dag()*L2))
            )

            LL += L1_sup + L2_sup

        RD += LL

        return RD

    # -------------------------------------------------
    # Propagation
    # -------------------------------------------------

    def evolve(self, rho0, tf, dt, options=None):

        if options is None:
            options = {"nsteps": 15000, "progress_bar": False}

        tlist = np.arange(0, tf + dt, dt) / 5308

        result = brmesolve(
            self.Hsys,
            rho0,
            tlist,
            a_ops=[[Q, self.calculate_DL] for Q in self.O_list],
            options=options
        )

        return result.states[-1]

    # -------------------------------------------------
    # Return all states
    # -------------------------------------------------

    def evolve_all(self, rho0, tf, dt, options=None):

        if options is None:
            options = {"nsteps": 15000, "progress_bar": False}

        tlist = np.arange(0, tf + dt, dt) / 5308

        result = brmesolve(
            self.Hsys,
            rho0,
            tlist,
            a_ops=[[Q, self.calculate_DL] for Q in self.O_list],
            options=options
        )

        return result.states

    # -------------------------------------------------
    # Initial density matrix
    # -------------------------------------------------

    def ground_state_density(self):

        return basis(self.nsite, 0) * basis(self.nsite, 0).dag()
