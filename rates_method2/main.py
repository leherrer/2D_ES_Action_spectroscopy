import numpy as np
import time

# OOP modules
from util_HAM_A import SystemHamiltonian
from util_RD import RedfieldPropagator
from util_2D_eigen import TwoDSpectra  # Your updated class using ThreadPoolExecutor


def main():

    # ==========================================================
    # Constants
    # ==========================================================
    kB = 0.69352      # cm^-1 / K
    hbar = 5308.8     # cm^-1 * fs

    # ==========================================================
    # Physical parameters
    # ==========================================================
    lam = 60.0        # cm^-1
    tau_c = 100.0     # fs
    T = 77.0          # K
    gamma = hbar / tau_c

    rate_rad =  np.array([0, 1/1e8, 1/1e6])  # Fluorescence rates
    rate_norad = np.array([0, 1/1e8, 1/100])  # Non-radiative rates

    # ==========================================================
    # Time parameters
    # ==========================================================
    t_final = 500.0
    dt = 10.0
    Time2s = [0]
    time_detection = 1e6

    # ==========================================================
    # Fourier windows
    # ==========================================================
    e1_range = (-400.0, 400.0, 5.0)
    e3_range = (-400.0, 400.0, 5.0)

    # ==========================================================
    # Build system Hamiltonian
    # ==========================================================
    J = -100.0
    energies = np.array([
        [0.0, -50.0, 8000.0],
        [0.0,  50.0, 8000.0]
    ])
    couplings = np.array([[0.0, J],[J,0.0]])
    K = np.copy(couplings)
    dipole_single = np.array([1.0, 1.0])
    dipole_x = np.array([1.0, -0.2])
    lam_single = np.array([1.0, 1.0])

    system = SystemHamiltonian(
        energies,
        couplings,
        K,
        dipole_single,
        dipole_x,
        lam_single,
        coupling_sites=[1,2],
        rate_rad=rate_rad, rate_norad=rate_norad
    )

    print("System dimension:", system.nsite)

    # ==========================================================
    # Build Redfield propagator
    # ==========================================================
    redfield = RedfieldPropagator(
        system,
        lam=lam,
        gamma=gamma,
        kT=kB*T
    )
    print("Redfield tensor dimension:", redfield.RD.shape)

    # ==========================================================
    # Build Liouville eigen engine
    # ==========================================================
    eigen_engine = TwoDSpectra(redfield, hbar)

    # ==========================================================
    # Compute 2D coherent spectra
    # ==========================================================
    start_time = time.time()
    Rrp, Rnr = eigen_engine.R_signal_para(
        Time2s, t_final, dt, Ncores=6
    )
    end_time = time.time()
    print(f"2D coherent spectra calculated in {end_time-start_time:.2f} s")

    # ==========================================================
    # Compute pathways of 2D coherent spectra
    # ==========================================================
    # start_time = time.time()
    # Rrp_path, Rnr_path, Phi1, Phi2, Phi3, Phi4, Phi5, Phi6 = eigen_engine.R_signal_para_pathway(
    #     Time2s, t_final, dt, Ncores=6
    # )
    # end_time = time.time()
    # print(f"2D coherent pathways calculated in {end_time-start_time:.2f} s")

    # ==========================================================
    # Compute 2D action spectra
    # ==========================================================
    start_time = time.time()
    Ract_rp, Ract_nr = eigen_engine.R_signal_para_action(
        Time2s, t_final, dt, time_detection, Ncores=6
    )
    end_time = time.time()
    print(f"2D action spectra calculated in {end_time-start_time:.2f} s")

    # # ==========================================================
    # # Compute pathways of 2D action spectra
    # # ==========================================================
    # start_time = time.time()
    # Ract_path_rp, Ract_path_nr, PhiA1, PhiA2, PhiA3, PhiA4, PhiA5, PhiA6, PhiA7, PhiA8 = eigen_engine.R_signal_para_action_pathway(
    #     Time2s, t_final, dt, time_detection, Ncores=6
    # )
    # end_time = time.time()
    # print(f"2D action pathways calculated in {end_time-start_time:.2f} s")

    # ==========================================================
    # Save spectra
    # ==========================================================
    np.save(f"Rrp_2D_coherent.npy", Rrp)
    np.save(f"Rnr_2D_coherent.npy", Rnr)

    #np.save(f"Rrp_2D_coherent_pathways.npy", Rrp_path)
    #np.save(f"Rnr_2D_coherent_pathways.npy", Rnr_path)
    #np.savez("Phi_2D_coherent_pathways.npz", Phi1=Phi1, Phi2=Phi2, Phi3=Phi3, Phi4=Phi4, Phi5=Phi5, Phi6=Phi6)

    np.save(f"Rrp_2D_action.npy", Ract_rp)
    np.save(f"Rnr_2D_action.npy", Ract_nr)

    #np.save(f"Rrp_2D_action_pathways.npy", Ract_path_rp)
    #np.save(f"Rnr_2D_action_pathways.npy", Ract_path_nr)
    #np.savez("Phi_2D_action_pathways.npz", Phi1=PhiA1, Phi2=PhiA2, Phi3=PhiA3, Phi4=PhiA4, Phi5=PhiA5, Phi6=PhiA6, Phi7=PhiA7, Phi8=PhiA8)

    print("All spectra and pathways saved.")


    # ==========================================================
    # Fourier transform for 2D coherent spectra
    # ==========================================================
    energy1s, energy3s, _, spectra_coh, times, _ = eigen_engine.Fourier_Transform(
        Rrp, Rnr,
        e1_range[0], e1_range[1], e1_range[2],
        e3_range[0], e3_range[1], e3_range[2],
        Time2s, t_final, dt
    )

    # Save 2D coherent spectra to txt
    for t2_idx, t2 in enumerate(Time2s):
        spectrum = spectra_coh[t2_idx]
        filename = f"2D_Coherent_t2-{t2:.1f}.txt"
        with open(filename, 'w') as f:
            for w1 in range(len(energy1s)):
                for w3 in range(len(energy3s)):
                    f.write(f"{energy1s[w1]:.8f} {energy3s[w3]:.8f} {spectrum[w3, w1]:.8f}\n")
                f.write("\n")
    print("2D coherent spectra written to txt files.")

    # ==========================================================
    # Fourier transform for 2D action spectra
    # ==========================================================
    energy1s, energy3s, _, spectra_act, times, _ = eigen_engine.Fourier_Transform(
        Ract_rp, Ract_nr,
        e1_range[0], e1_range[1], e1_range[2],
        e3_range[0], e3_range[1], e3_range[2],
        Time2s, t_final, dt
    )

    # Save 2D action spectra to txt
    for t2_idx, t2 in enumerate(Time2s):
        spectrum = spectra_act[t2_idx]
        filename = f"2D_Action_t2-{t2:.1f}.txt"
        with open(filename, 'w') as f:
            for w1 in range(len(energy1s)):
                for w3 in range(len(energy3s)):
                    f.write(f"{energy1s[w1]:.8f} {energy3s[w3]:.8f} {spectrum[w3, w1]:.8f}\n")
                f.write("\n")
    print("2D action spectra written to txt files.")


# ==========================================================
# Entry point protection
# ==========================================================
if __name__ == "__main__":
    main()
