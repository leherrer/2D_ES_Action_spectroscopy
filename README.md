# 2D Spectroscopy Action Simulation with Redfield Dynamics

This repository provides a Python implementation for simulating **2D coherent and action spectra** of molecular systems using **Redfield theory** and **Liouville-space eigen decomposition**. It includes computation of total spectra as well as individual Liouville pathways.

---

## Features

* Compute **2D coherent spectra** (rephasing and non-rephasing signals).
* Compute **2D action spectra** for third-order response functions.
* Track contributions of **individual Liouville pathways** (`Phi1, Phi2, ...`).
* Parallelized computation using **threads** for large-scale simulations.
* Fourier transform of time-domain signals to frequency-domain spectra.
* Save spectra to **.txt** and **.npy** files for further analysis.

---

## Requirements

* Python 3.8+
* NumPy
* SciPy
* QuTiP (`qutip` for quantum operators)
* Optional: `matplotlib` for plotting


---

## Repository Structure

```
.
├── util_HAM_A.py         # Defines the system Hamiltonian
├── util_RD.py            # Redfield propagator construction
├── util_2D_eigen.py     # 2D spectroscopy engine (TwoDSpectra class)
├── main.py               # Example script to run spectra calculations
├── README.md             # This file
└── results/             # Folder to save output spectra
```

---

## Usage

### 1. Prepare the system Hamiltonian

```python
from util_HAM_A import SystemHamiltonian

system = SystemHamiltonian(
    energies,
    couplings,
    K,
    dipole_single,
    dipole_x,
    lam_single,
    coupling_sites=[1,2]
)
```

### 2. Build the Redfield propagator

```python
from util_RD import RedfieldPropagator

redfield = RedfieldPropagator(
    system,
    lam=60.0,
    gamma=5308.8/100.0,
    kT=0.69352*77.0,
    rate_rad=np.array([0, 1/1e8, 1/1e6]), rate_norad=np.array([0, 1/1e8, 1/100]))
)
```

### 3. Initialize 2D spectroscopy engine

```python
from util_2D_eigen import TwoDSpectra

eigen_engine = TwoDSpectra(redfield, hbar=5308.8)
```

### 4. Compute 2D coherent spectra

```python
Rrp, Rnr = eigen_engine.R_signal_para(Time2s=[0], time_final=500, dt=10, Ncores=6)
```

### 5. Compute 2D action spectra

```python
Ract_rp, Ract_nr = eigen_engine.R_signal_para_action(Time2s=[0], time_final=500, dt=10, time_detection=1e6, Ncores=6)
```

### 6. Compute pathways (optional)

```python
Rrp_path, Rnr_path, Phi1, Phi2, Phi3, Phi4, Phi5, Phi6 = eigen_engine.R_signal_para_pathway(
    Time2s=[0], time_final=500, dt=10, time_detection=1e6, Ncores=6
)
```

### 7. Fourier transform and save spectra

```python
energy1s, energy3s, t2s, spectra, times, _ = eigen_engine.Fourier_Transform(
    Rrp, Rnr,
    e1_min=-400, e1_max=400, de1=5,
    e3_min=-400, e3_max=400, de3=5,
    time2s=[0],
    time_final=500,
    dt=10
)

# Save each t2 slice
for t2_idx, t2 in enumerate(t2s):
    spectrum = spectra[t2_idx]
    filename = f"2D_Coherent_t2-{t2:.1f}.txt"
    with open(filename, 'w') as f:
        for w1 in range(len(energy1s)):
            for w3 in range(len(energy3s)):
                f.write(f"{energy1s[w1]:.8f} {energy3s[w3]:.8f} {spectrum[w3, w1]:.8f}\n")
            f.write("\n")
```

---

## Notes

* All time-domain propagations are parallelized using **threads** (`ThreadPoolExecutor`) for performance.
* The code supports **arbitrary system size**, but memory usage scales with `N^2` due to Liouville-space representation.
* Adjust **`time_final`** and **`dt`** to control spectral resolution.

---

## References


* QuTiP: Quantum Toolbox in Python, [http://qutip.org](http://qutip.org)

---

## License

This repository is provided under the MIT License. Free to use, modify, and distribute.

