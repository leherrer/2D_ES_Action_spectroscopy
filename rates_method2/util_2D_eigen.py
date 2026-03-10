import numpy as np
import scipy.linalg as sp
from numpy import linalg as LA
from qutip import *
from concurrent.futures import ThreadPoolExecutor


class TwoDSpectra:

    def __init__(self, redfield, hbar):

        self.hbar = hbar

        # System dimension
        self.N = redfield.nsite

        # Liouvillian matrix
        L = redfield.RD.full(order="C")

        # Diagonalize
        evals, evecs = LA.eig(L)

        self.U = evecs
        self.U_1 = LA.inv(evecs)
        self.diagonal_matrix = np.diag(evals)

        # Dipoles
        mu_plus = Qobj(redfield.mu_p)
        mu_minus = Qobj(redfield.mu_m)

        # Superoperators
        self.mu_plus_s = (spre(mu_plus) - spost(mu_plus)).full(order="C")
        self.mu_minus_s = (spre(mu_minus) - spost(mu_minus)).full(order="C")

        self.mu_plus_s_left = spre(mu_plus).full(order="C")
        self.mu_plus_s_right = spost(mu_plus).full(order="C")

        self.mu_minus_s_left = spre(mu_minus).full(order="C")
        self.mu_minus_s_right = spost(mu_minus).full(order="C")

        # Initial density matrix
        rho0 = redfield.ground_state_density()

        self.vec_rho0 = rho0.full(order="C").reshape(self.N**2)

        # detection vector
        self.mu_plus_vec_left = (
            mu_plus.dag().full(order="C")
        ).reshape(self.N**2)

        ##Projectors vectors

        self.P0_vec_left = (redfield.P0.dag().full(order="C")).reshape(self.N**2, order="C")
        self.P1_vec_left = (redfield.P1.dag().full(order="C")).reshape(self.N**2, order="C")
        self.P2_vec_left = (redfield.P2.dag().full(order="C")).reshape(self.N**2, order="C")

        self.P_T_vec_left_exp = self.P1_vec_left + self.P2_vec_left




    # ------------------------------------------------
    # single response element
    # ------------------------------------------------

    def Resp_para(self, i, j, k, t1, t2, t3):

        U = self.U
        U_1 = self.U_1
        D = self.diagonal_matrix
        hbar = self.hbar

        rp = (
            self.mu_plus_vec_left
            @ U @ sp.expm(D*t3/hbar) @ U_1
            @ self.mu_plus_s
            @ U @ sp.expm(D*t2/hbar) @ U_1
            @ self.mu_plus_s
            @ U @ sp.expm(D*t1/hbar) @ U_1
            @ self.mu_minus_s
            @ self.vec_rho0
        )

        nr = (
            self.mu_plus_vec_left
            @ U @ sp.expm(D*t3/hbar) @ U_1
            @ self.mu_plus_s
            @ U @ sp.expm(D*t2/hbar) @ U_1
            @ self.mu_minus_s
            @ U @ sp.expm(D*t1/hbar) @ U_1
            @ self.mu_plus_s
            @ self.vec_rho0
        )

        return i, j, k, rp, nr

    # ------------------------------------------------
    # pathways
    # ------------------------------------------------

    def Resp_para_pathways(self, i, j, k, t1, t2, t3):

        U = self.U
        U_1 = self.U_1
        D = self.diagonal_matrix
        hbar = self.hbar

        ## Rephasing
        phi1 = (
            self.mu_plus_vec_left
            @ U @ sp.expm(D*t3/hbar) @ U_1
            @ self.mu_plus_s_right
            @ U @ sp.expm(D*t2/hbar) @ U_1
            @ self.mu_plus_s_left
            @ U @ sp.expm(D*t1/hbar) @ U_1
            @ self.mu_minus_s_right
            @ self.vec_rho0
        )

        phi2 = (
            self.mu_plus_vec_left
            @ U @ sp.expm(D*t3/hbar) @ U_1
            @ self.mu_plus_s_left
            @ U @ sp.expm(D*t2/hbar) @ U_1
            @ self.mu_plus_s_right
            @ U @ sp.expm(D*t1/hbar) @ U_1
            @ self.mu_minus_s_right
            @ self.vec_rho0
        )

        phi3 = (
            self.mu_plus_vec_left
            @ U @ sp.expm(D*t3/hbar) @ U_1
            @ self.mu_plus_s_left
            @ U @ sp.expm(D*t2/hbar) @ U_1
            @ self.mu_plus_s_left
            @ U @ sp.expm(D*t1/hbar) @ U_1
            @ self.mu_minus_s_right
            @ self.vec_rho0
        )

        ## Non-rephasing
        phi4 = (
            self.mu_plus_vec_left
            @ U @ sp.expm(D*t3/hbar) @ U_1
            @ self.mu_plus_s_right
            @ U @ sp.expm(D*t2/hbar) @ U_1
            @ self.mu_minus_s_right
            @ U @ sp.expm(D*t1/hbar) @ U_1
            @ self.mu_plus_s_left
            @ self.vec_rho0
        )

        phi5 = (
            self.mu_plus_vec_left
            @ U @ sp.expm(D*t3/hbar) @ U_1
            @ self.mu_plus_s_left
            @ U @ sp.expm(D*t2/hbar) @ U_1
            @ self.mu_minus_s_left
            @ U @ sp.expm(D*t1/hbar) @ U_1
            @ self.mu_plus_s_left
            @ self.vec_rho0
        )

        phi6 = (
            self.mu_plus_vec_left
            @ U @ sp.expm(D*t3/hbar) @ U_1
            @ self.mu_plus_s_left
            @ U @ sp.expm(D*t2/hbar) @ U_1
            @ self.mu_minus_s_right
            @ U @ sp.expm(D*t1/hbar) @ U_1
            @ self.mu_plus_s_left
            @ self.vec_rho0
        )

        rp = phi1 + phi2 - phi3
        nr = phi4 + phi5 - phi6

        return i, j, k, rp, nr, phi1, phi2, phi3, phi4, phi5, phi6

    # ------------------------------------------------
    # compute response
    # ------------------------------------------------

    def R_signal_para(self, time2s, time_final, dt, Ncores):

        times = np.arange(0.0, time_final, dt)

        Rrp = np.zeros((len(times), len(time2s), len(times)), dtype=complex)
        Rnr = np.zeros_like(Rrp)

        with ThreadPoolExecutor(max_workers=Ncores) as executor:

            futures = []

            for i, t3 in enumerate(times):
                for j, t2 in enumerate(time2s):
                    for k, t1 in enumerate(times):

                        futures.append(
                            executor.submit(
                                self.Resp_para,
                                i, j, k, t1, t2, t3
                            )
                        )

            for future in futures:

                i, j, k, rp, nr = future.result()

                Rrp[i, j, k] = rp
                Rnr[i, j, k] = nr

        return Rrp, Rnr

    # ------------------------------------------------
    # compute response + pathways
    # ------------------------------------------------

    def R_signal_para_pathway(self, time2s, time_final, dt, Ncores):

        times = np.arange(0.0, time_final, dt)

        shape = (len(times), len(time2s), len(times))

        Rrp = np.zeros(shape, dtype=complex)
        Rnr = np.zeros(shape, dtype=complex)

        Phi1 = np.zeros(shape, dtype=complex)
        Phi2 = np.zeros(shape, dtype=complex)
        Phi3 = np.zeros(shape, dtype=complex)
        Phi4 = np.zeros(shape, dtype=complex)
        Phi5 = np.zeros(shape, dtype=complex)
        Phi6 = np.zeros(shape, dtype=complex)

        with ThreadPoolExecutor(max_workers=Ncores) as executor:

            futures = []

            for i, t3 in enumerate(times):
                for j, t2 in enumerate(time2s):
                    for k, t1 in enumerate(times):

                        futures.append(
                            executor.submit(
                                self.Resp_para_pathways,
                                i, j, k, t1, t2, t3
                            )
                        )

            for future in futures:

                (
                    i, j, k,
                    rp, nr,
                    phi1, phi2, phi3,
                    phi4, phi5, phi6
                ) = future.result()

                Rrp[i, j, k] = rp
                Rnr[i, j, k] = nr

                Phi1[i, j, k] = phi1
                Phi2[i, j, k] = phi2
                Phi3[i, j, k] = phi3
                Phi4[i, j, k] = phi4
                Phi5[i, j, k] = phi5
                Phi6[i, j, k] = phi6

        return Rrp, Rnr, Phi1, Phi2, Phi3, Phi4, Phi5, Phi6

    ###################
    # Action ##
    ##################

    def rho_four_order_action(self, t1, t2, t3):

        U = self.U
        U_1 = self.U_1
        D = self.diagonal_matrix
        hbar = self.hbar

        rho_rp = (
            self.mu_minus_s
            @ U @ sp.expm(D*t3/hbar) @ U_1
            @ self.mu_plus_s
            @ U @ sp.expm(D*t2/hbar) @ U_1
            @ self.mu_plus_s
            @ U @ sp.expm(D*t1/hbar) @ U_1
            @ self.mu_minus_s
            @ self.vec_rho0
        )

        rho_nr = (
            self.mu_minus_s
            @ U @ sp.expm(D*t3/hbar) @ U_1
            @ self.mu_plus_s
            @ U @ sp.expm(D*t2/hbar) @ U_1
            @ self.mu_minus_s
            @ U @ sp.expm(D*t1/hbar) @ U_1
            @ self.mu_plus_s
            @ self.vec_rho0
        )

        return rho_rp, rho_nr

    def rho_four_order_actionpathways(self, t1, t2, t3):

        U = self.U
        U_1 = self.U_1
        D = self.diagonal_matrix
        hbar = self.hbar

        phi1 = (
            self.mu_minus_s_right
            @ U @ sp.expm(D*t3/hbar) @ U_1
            @ self.mu_plus_s_right
            @ U @ sp.expm(D*t2/hbar) @ U_1
            @ self.mu_plus_s_left
            @ U @ sp.expm(D*t1/hbar) @ U_1
            @ self.mu_minus_s_right
            @ self.vec_rho0
        )

        phi2 = (
            self.mu_minus_s_right
            @ U @ sp.expm(D*t3/hbar) @ U_1
            @ self.mu_plus_s_left
            @ U @ sp.expm(D*t2/hbar) @ U_1
            @ self.mu_plus_s_right
            @ U @ sp.expm(D*t1/hbar) @ U_1
            @ self.mu_minus_s_right
            @ self.vec_rho0
        )

        phi3 = (
            self.mu_minus_s_left
            @ U @ sp.expm(D*t3/hbar) @ U_1
            @ self.mu_plus_s_left
            @ U @ sp.expm(D*t2/hbar) @ U_1
            @ self.mu_plus_s_left
            @ U @ sp.expm(D*t1/hbar) @ U_1
            @ self.mu_minus_s_right
            @ self.vec_rho0
        )

        phi4 = (
            self.mu_minus_s_right
            @ U @ sp.expm(D*t3/hbar) @ U_1
            @ self.mu_plus_s_left
            @ U @ sp.expm(D*t2/hbar) @ U_1
            @ self.mu_plus_s_left
            @ U @ sp.expm(D*t1/hbar) @ U_1
            @ self.mu_minus_s_right
            @ self.vec_rho0
        )

        phi5 = (
            self.mu_minus_s_right
            @ U @ sp.expm(D*t3/hbar) @ U_1
            @ self.mu_plus_s_right
            @ U @ sp.expm(D*t2/hbar) @ U_1
            @ self.mu_minus_s_right
            @ U @ sp.expm(D*t1/hbar) @ U_1
            @ self.mu_plus_s_left
            @ self.vec_rho0
        )

        phi6 = (
            self.mu_minus_s_right
            @ U @ sp.expm(D*t3/hbar) @ U_1
            @ self.mu_plus_s_left
            @ U @ sp.expm(D*t2/hbar) @ U_1
            @ self.mu_minus_s_left
            @ U @ sp.expm(D*t1/hbar) @ U_1
            @ self.mu_plus_s_left
            @ self.vec_rho0
        )

        phi7 = (
            self.mu_minus_s_left
            @ U @ sp.expm(D*t3/hbar) @ U_1
            @ self.mu_plus_s_left
            @ U @ sp.expm(D*t2/hbar) @ U_1
            @ self.mu_minus_s_right
            @ U @ sp.expm(D*t1/hbar) @ U_1
            @ self.mu_plus_s_left
            @ self.vec_rho0
        )

        phi8 = (
            self.mu_minus_s_right
            @ U @ sp.expm(D*t3/hbar) @ U_1
            @ self.mu_plus_s_left
            @ U @ sp.expm(D*t2/hbar) @ U_1
            @ self.mu_minus_s_right
            @ U @ sp.expm(D*t1/hbar) @ U_1
            @ self.mu_plus_s_left
            @ self.vec_rho0
        )

        rp = phi1 + phi2 + phi3 - phi4
        nr = phi5 + phi6 + phi7 - phi8

        return -rp, -nr, phi1, phi2, phi3, phi4, phi5, phi6, phi7, phi8


    def detection_time_grid(self, tdetmax):

        points_per_decade = 20
        decades = np.log10(tdetmax)

        n_points = int(points_per_decade * decades)

        times = np.array([0])
        times = np.concatenate(
            (times, np.logspace(0, np.log10(tdetmax), n_points))
        )

        return times

    def Resp_para_RD_action(self, i, j, k, t1, t2, t3, tdetmax):

        Time_detection = self.detection_time_grid(tdetmax)

        RP = np.zeros(len(Time_detection), dtype=complex)
        NR = np.zeros(len(Time_detection), dtype=complex)

        rho_rp, rho_nr = self.rho_four_order_action(t1, t2, t3)

        for l, td in enumerate(Time_detection):

            aux_rp = (
                self.P_T_vec_left_exp
                @ self.U @ sp.expm(self.diagonal_matrix*td/self.hbar)
                @ self.U_1
                @ rho_rp
            )

            aux_nr = (
                self.P_T_vec_left_exp
                @ self.U @ sp.expm(self.diagonal_matrix*td/self.hbar)
                @ self.U_1
                @ rho_nr
            )

            RP[l] = aux_rp
            NR[l] = aux_nr

        from scipy import integrate

        rp = integrate.simpson(RP, x=Time_detection)
        nr = integrate.simpson(NR, x=Time_detection)

        return i, j, k, rp, nr


    def Resp_para_RD_action_integral(self, t1, t2, t3, tdetmax):

        Time_detection = self.detection_time_grid(tdetmax)

        RP = np.zeros(len(Time_detection), dtype=complex)
        NR = np.zeros(len(Time_detection), dtype=complex)

        rho_rp, rho_nr = self.rho_four_order_action(t1, t2, t3)

        for l, td in enumerate(Time_detection):

            RP[l] = (
                self.P_T_vec_left_exp
                @ self.U @ sp.expm(self.diagonal_matrix*td/self.hbar)
                @ self.U_1
                @ rho_rp
            )

            NR[l] = (
                self.P_T_vec_left_exp
                @ self.U @ sp.expm(self.diagonal_matrix*td/self.hbar)
                @ self.U_1
                @ rho_nr
            )

        from scipy import integrate

        rp = integrate.simpson(RP, x=Time_detection)
        nr = integrate.simpson(NR, x=Time_detection)

        return rp, nr, RP, NR, Time_detection


    def Resp_para_RD_action_pathways(self, i, j, k, t1, t2, t3, tdetmax):

        from scipy import integrate
        import scipy.linalg as sp

        Time_detection = self.detection_time_grid(tdetmax)

        N_points = len(Time_detection)

        RP = np.zeros(N_points, dtype=complex)
        NR = np.zeros(N_points, dtype=complex)

        Phi1 = np.zeros(N_points, dtype=complex)
        Phi2 = np.zeros(N_points, dtype=complex)
        Phi3 = np.zeros(N_points, dtype=complex)
        Phi4 = np.zeros(N_points, dtype=complex)
        Phi5 = np.zeros(N_points, dtype=complex)
        Phi6 = np.zeros(N_points, dtype=complex)
        Phi7 = np.zeros(N_points, dtype=complex)
        Phi8 = np.zeros(N_points, dtype=complex)

        (
            rho_rp,
            rho_nr,
            rho_phi1,
            rho_phi2,
            rho_phi3,
            rho_phi4,
            rho_phi5,
            rho_phi6,
            rho_phi7,
            rho_phi8
        ) = self.rho_four_order_actionpathways(t1, t2, t3)

        for l, td in enumerate(Time_detection):

            propagator = (
                self.U
                @ sp.expm(self.diagonal_matrix * td / self.hbar)
                @ self.U_1
            )

            RP[l] = self.P_T_vec_left_exp @ propagator @ rho_rp
            NR[l] = self.P_T_vec_left_exp @ propagator @ rho_nr

            Phi1[l] = self.P_T_vec_left_exp @ propagator @ rho_phi1
            Phi2[l] = self.P_T_vec_left_exp @ propagator @ rho_phi2
            Phi3[l] = self.P_T_vec_left_exp @ propagator @ rho_phi3
            Phi4[l] = self.P_T_vec_left_exp @ propagator @ rho_phi4

            Phi5[l] = self.P_T_vec_left_exp @ propagator @ rho_phi5
            Phi6[l] = self.P_T_vec_left_exp @ propagator @ rho_phi6
            Phi7[l] = self.P_T_vec_left_exp @ propagator @ rho_phi7
            Phi8[l] = self.P_T_vec_left_exp @ propagator @ rho_phi8

        rp = integrate.simpson(RP, x=Time_detection)
        nr = integrate.simpson(NR, x=Time_detection)

        phi1 = integrate.simpson(Phi1, x=Time_detection)
        phi2 = integrate.simpson(Phi2, x=Time_detection)
        phi3 = integrate.simpson(Phi3, x=Time_detection)
        phi4 = integrate.simpson(Phi4, x=Time_detection)
        phi5 = integrate.simpson(Phi5, x=Time_detection)
        phi6 = integrate.simpson(Phi6, x=Time_detection)
        phi7 = integrate.simpson(Phi7, x=Time_detection)
        phi8 = integrate.simpson(Phi8, x=Time_detection)

        return (
            i, j, k,
            rp, nr,
            phi1, phi2, phi3, phi4,
            phi5, phi6, phi7, phi8
        )


    def R_signal_para_action(self, time2s, time_final, dt, tdetmax ,Ncores):
        """
        Compute the third-order 2D action response in parallel.
        Returns the total RP and NR signals.
        """
        times = np.arange(0.0, time_final, dt)
        N_t = len(times)
        N_t2 = len(time2s)

        print(f"--- Spectrum will require O({N_t*N_t2*N_t}) propagations.")
        print("--- Calculating third-order response function ...")

        Rsignal_rp = np.zeros((N_t, N_t2, N_t), dtype=complex)
        Rsignal_nr = np.zeros((N_t, N_t2, N_t), dtype=complex)

        with ThreadPoolExecutor(max_workers=Ncores) as executor:
            futures = []
            for i, t3 in enumerate(times):
                for j, t2 in enumerate(time2s):
                    for k, t1 in enumerate(times):
                        futures.append(
                            executor.submit(self.Resp_para_RD_action, i, j, k, t1, t2, t3, tdetmax)
                        )

            for future in futures:
                i, j, k, rp, nr = future.result()
                Rsignal_rp[i, j, k] = rp
                Rsignal_nr[i, j, k] = nr

        return Rsignal_rp, Rsignal_nr


    def R_signal_para_action_pathway(self, time2s, time_final, dt, tdetmax , Ncores):
        """
        Compute the third-order 2D action response in parallel, including
        the 8 individual Liouville pathways for monitoring contributions.
        """
        times = np.arange(0.0, time_final, dt)
        N_t = len(times)
        N_t2 = len(time2s)

        print(f"--- Spectrum will require O({N_t*N_t2*N_t}) propagations.")
        print("--- Calculating third-order response function ...")

        # Initialize arrays
        Rsignal_rp = np.zeros((N_t, N_t2, N_t), dtype=complex)
        Rsignal_nr = np.zeros((N_t, N_t2, N_t), dtype=complex)
        Phi = [np.zeros((N_t, N_t2, N_t), dtype=complex) for _ in range(8)]

        print('PT_vector', self.P_T_vec_left_exp)  # For debug

        with ThreadPoolExecutor(max_workers=Ncores) as executor:
            futures = []
            for i, t3 in enumerate(times):
                for j, t2 in enumerate(time2s):
                    for k, t1 in enumerate(times):
                        futures.append(
                            executor.submit(
                                self.Resp_para_RD_action_pathways, i, j, k, t1, t2, t3, tdetmax
                            )
                        )

            for future in futures:
                result = future.result()
                i, j, k, rp, nr, *phi_vals = result
                Rsignal_rp[i, j, k] = rp
                Rsignal_nr[i, j, k] = nr
                for n, phi_val in enumerate(phi_vals):
                    Phi[n][i, j, k] = phi_val

        return (Rsignal_rp, Rsignal_nr, *Phi)


    def Fourier_Transform(self,
                      Rrp, Rnr,
                      e1_min, e1_max, de1,
                      e3_min, e3_max, de3,
                      time2s,
                      time_final, dt):

        energy1s = np.arange(e1_min, e1_max, de1)
        energy3s = np.arange(e3_min, e3_max, de3)

        omega1s = energy1s
        omega3s = energy3s

        times = np.arange(0.0, time_final, dt)

        spectrum = np.zeros((len(omega3s), len(time2s), len(omega1s)))

        Rsignal = [Rrp, Rnr]

        # Fourier kernels
        expi1 = np.exp(1j*(1/self.hbar)*np.outer(omega1s, times))
        expi1[:,0] *= 0.5*dt
        expi1[:,1:] *= dt

        expi3 = np.exp(1j*(1/self.hbar)*np.outer(omega3s, times))
        expi3[:,0] *= 0.5*dt
        expi3[:,1:] *= dt

        # non-rephasing
        spectrum = np.einsum(
            'ws,xu,uts->xtw',
            expi1,
            expi3,
            Rnr
        ).real

        # rephasing
        spectrum += np.einsum(
            'ws,xu,uts->xtw',
            expi1.conj(),
            expi3,
            Rrp
        ).real

        spectra = [spectrum[:, t2, :] for t2 in range(len(time2s))]

        print("done.")

        return energy1s, energy3s, time2s, spectra, times, Rsignal


    def Fourier_Transform_rp(self,
                         Rrp,
                         e1_min, e1_max, de1,
                         e3_min, e3_max, de3,
                         time2s,
                         time_final, dt):

        energy1s = np.arange(e1_min, e1_max, de1)
        energy3s = np.arange(e3_min, e3_max, de3)

        omega1s = energy1s
        omega3s = energy3s

        times = np.arange(0.0, time_final, dt)

        expi1 = np.exp(1j*(1/self.hbar)*np.outer(omega1s, times))
        expi1[:,0] *= 0.5*dt
        expi1[:,1:] *= dt

        expi3 = np.exp(1j*(1/self.hbar)*np.outer(omega3s, times))
        expi3[:,0] *= 0.5*dt
        expi3[:,1:] *= dt

        spectrum = np.einsum(
            'ws,xu,uts->xtw',
            expi1.conj(),
            expi3,
            Rrp
        ).real

        spectra = [spectrum[:, t2, :] for t2 in range(len(time2s))]

        print("done.")

        return energy1s, energy3s, time2s, spectra, times, [Rrp]

    def Fourier_Transform_nr(self,
                         Rnr,
                         e1_min, e1_max, de1,
                         e3_min, e3_max, de3,
                         time2s,
                         time_final, dt):

        energy1s = np.arange(e1_min, e1_max, de1)
        energy3s = np.arange(e3_min, e3_max, de3)

        omega1s = energy1s
        omega3s = energy3s

        times = np.arange(0.0, time_final, dt)

        expi1 = np.exp(1j*(1/self.hbar)*np.outer(omega1s, times))
        expi1[:,0] *= 0.5*dt
        expi1[:,1:] *= dt

        expi3 = np.exp(1j*(1/self.hbar)*np.outer(omega3s, times))
        expi3[:,0] *= 0.5*dt
        expi3[:,1:] *= dt

        spectrum = np.einsum(
            'ws,xu,uts->xtw',
            expi1,
            expi3,
            Rnr
        ).real

        spectra = [spectrum[:, t2, :] for t2 in range(len(time2s))]

        print("done.")

        return energy1s, energy3s, time2s, spectra, times, [Rnr]
