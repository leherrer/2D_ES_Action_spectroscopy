import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pylab import cm
from PIL import Image

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'Latin Modern Roman'
plt.rcParams['font.size'] = 20
plt.rcParams['axes.linewidth'] = 2


def print_max_peaks(X_grid, Y_grid, Z_grid, N=4, min_dist=100):
    Zflat = Z_grid.flatten()
    idx_sorted = np.argsort(Zflat)[::-1]

    peaks = []

    for idx in idx_sorted:
        j = idx // Z_grid.shape[1]
        i = idx % Z_grid.shape[1]

        x_peak = X_grid[j, i]
        y_peak = Y_grid[j, i]
        z_peak = Z_grid[j, i]

        too_close = False
        for xp, yp, zp in peaks:
            dist = np.sqrt((x_peak - xp)**2 + (y_peak - yp)**2)
            if dist < min_dist:
                too_close = True
                break

        if not too_close:
            peaks.append((x_peak, y_peak, z_peak))
            print(f"Peak {len(peaks)}: x = {x_peak:.3f}, y = {y_peak:.3f}, z = {z_peak:.6f}")

        if len(peaks) >= N:
            break


def add_manual_squares(ax, positions, labels):
    for (x, y), text in zip(positions, labels):
        ax.scatter(x, y, s=300, marker='s',
                   facecolor='none', edgecolor='black', linewidth=2)

        ax.text(x + 25, y + 25, text,
                fontsize=34, fontweight='bold', color='black')


def plot_single_contour(file_path,
                        output_path="single_plot.jpg",
                        plot_title="",
                        add_squares=False):

    fig, ax = plt.subplots(figsize=(12, 10))

    # Load data
    data = np.loadtxt(file_path)
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]

    # Normalize by max absolute value
    max_abs_z = np.max(np.abs(z))
    if max_abs_z != 0:
        z_normalized = z / max_abs_z
    else:
        z_normalized = z  # all zeros


    # Create meshgrid
    X_unique = np.unique(x)
    Y_unique = np.unique(y)
    X_grid, Y_grid = np.meshgrid(X_unique, Y_unique)

    Z_grid = np.zeros_like(X_grid)
    for i, xi in enumerate(X_unique):
        for j, yi in enumerate(Y_unique):
            Z_grid[j, i] = z_normalized[(x == xi) & (y == yi)][0]

    # Contour plot
    cp = ax.contourf(
        X_grid, Y_grid, Z_grid,
        cmap='seismic',
        levels=np.linspace(-1, 1, 100)  # symmetric for positives and negatives
    )

    ax.contour(
        X_grid, Y_grid, Z_grid,
        levels=np.linspace(-1, 1, 10),
        colors='black', linewidths=1
    )

    # Diagonal
    ax.plot([X_unique[0], X_unique[-1]],
            [Y_unique[0], Y_unique[-1]],
            linestyle='--', color='black', linewidth=2)

    # Labels
    ax.set_xlabel(r'$\omega_1\ (\mathrm{cm}^{-1})$', fontsize=32, fontweight='bold')
    ax.set_ylabel(r'$\omega_3\ (\mathrm{cm}^{-1})$', fontsize=32, fontweight='bold')

    # Ticks
    #ticks = np.arange(-500, 501, 100)
    #ax.set_xticks(ticks)
    #ax.set_yticks(ticks)

    ax.tick_params(axis='both', which='major',
                   labelsize=28, width=2, direction='in', length=6)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    # Title inside plot
    if plot_title:
        ax.text(
            0.5, 0.92, plot_title,
            ha='center', va='bottom',
            fontsize=28, fontweight='bold',
            transform=ax.transAxes,
            bbox=dict(facecolor='white',
                      edgecolor='black',
                      boxstyle='round,pad=0.3',
                      alpha=0.8)
        )

    # Optional peak printing + manual squares
    if add_squares:
        print_max_peaks(X_grid, Y_grid, Z_grid, N=4)

        square_positions = [
            (-145, -155),
            (95, 100),
            (-145, 105),
            (95, -155),
        ]
        square_labels = ["DP1", "DP2", "CP12", "CP21"]

        add_manual_squares(ax, square_positions, square_labels)

    # Colorbar
    cbar = fig.colorbar(cp)
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    cbar.ax.tick_params(labelsize=24, width=2)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight('bold')

    # Save
    plt.tight_layout()
    plt.savefig("temp.png", dpi=300, bbox_inches='tight')
    image = Image.open("temp.png")
    image.convert("RGB").save(output_path, "JPEG", quality=90, optimize=True)

    print(f"Saved figure as {output_path}")


#path  = "2d_t2-0.0_HEOM_dt-10_tf-500_L-5_K-1_g-100.0_l-20.0_J--20.dat"
#path = "2D_Coherent_t2-0.0.txt"
path = "2D_Action_t2-0.0.txt"

#output_path = "2D_Coherent_t2-0.0.jpg"
output_path = "2D_Action_t2-0.0.jpg"

plot_single_contour(
        file_path=path,
        output_path=output_path,
        plot_title=r"$t_2 = 0$ fs",
        add_squares=False
    )
