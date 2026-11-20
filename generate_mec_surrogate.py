import time
import numpy as np
import scipy.io as sio
from scipy.interpolate import interp1d

# =========================================================================
# GENERATE RECTANGULAR MEC SURROGATE LOOKUP TABLES (mec_rectangular_surrogate.mat)
# =========================================================================
# Solves the nonlinear Magnetic Equivalent Circuit (MEC) equations from
# Lusine.m across an (air_gap, coil_current) grid and exports the lookup
# tables required by StaticMEC.py and DynamicMEC.py.
# =========================================================================

start_time = time.time()

# 1. Electromagnet geometry (exact dimensions from Lusine.m) [m]
r1 = 10.45e-3       # Inner core radius
r2 = 25.40e-3       # Coil outer / outer shell inner radius
r3 = 27.55e-3       # Outer shell radius
b_ld = 6.30e-3      # Suspended load disk thickness
b_bs = 7.30e-3      # Bottom base plate thickness
l_cr = 29.50e-3     # Core and shell axial height
mu_0 = 4.0 * np.pi * 1e-7  # Permeability of free space [H/m]

# 2. Electromechanical parameters and constraints
coil_turns = 300        # Number of winding turns [-]
coil_resistance = 2.5   # Coil electrical resistance [Ohm]
load_mass = 0.5         # Suspended load mass [kg]
max_voltage = 24.0      # Maximum supply voltage [V]
max_current = 10.0      # Maximum coil current [A]
min_air_gap = 0.05e-3   # Minimum mechanical air-gap stop [m]
max_air_gap = 2.00e-3   # Maximum mechanical air-gap stop [m]


def iron_bh_curve(flux_density_B):
    """Nonlinear iron magnetization curve H = f(B) with saturation at 1.9 T."""
    if flux_density_B < 0.0:
        return 0.0
    elif flux_density_B < 1.9:
        return -1818.18 * np.log(1.0 - flux_density_B / 1.9)
    else:
        return 10000.0 + flux_density_B


def compute_residuals(x, n, R_d, Fi_ld, S_ld, S_ldv1, S_ldv2, S_bsv1, S_bsv2,
                      S_cr, S_cd, S_bs, l_s_crk, l_s_ld, l_s_ldv, l_s_bs, l_s_bsv,
                      R_delta1, R_delta2):
    """Evaluate Kirchhoff magnetic loop and node equations f(x) = 0."""
    f = np.zeros(2 * n + 1)

    # Top magnetic loop (across air gaps, suspended load, and top core slice)
    f[2 * n] = (
        Fi_ld * R_delta1 + Fi_ld * R_delta2
        + iron_bh_curve(Fi_ld / S_ld) * l_s_ld
        + iron_bh_curve(Fi_ld / S_ldv1) * l_s_ldv
        + iron_bh_curve(Fi_ld / S_ldv2) * l_s_ldv
        + iron_bh_curve(Fi_ld / S_cr) * l_s_crk
        + iron_bh_curve(Fi_ld / S_cd) * l_s_crk
        - x[0] * R_d - x[2 * n]
    )

    # Intermediate (n - 1) magnetic loops along the coil window
    for i in range(n - 1):
        f[i] = (
            iron_bh_curve(x[n + i] / S_cr) * l_s_crk
            + iron_bh_curve(x[n + i] / S_cd) * l_s_crk
            + x[i] * R_d - x[i + 1] * R_d - x[2 * n]
        )

    # Bottom magnetic loop (includes bottom iron base plate)
    f[n - 1] = (
        iron_bh_curve(x[2 * n - 1] / S_cr) * l_s_crk
        + iron_bh_curve(x[2 * n - 1] / S_cd) * l_s_crk
        + iron_bh_curve(x[2 * n - 1] / S_bs) * l_s_bs
        + iron_bh_curve(x[2 * n - 1] / S_bsv1) * l_s_bsv
        + iron_bh_curve(x[2 * n - 1] / S_bsv2) * l_s_bsv
        + x[n - 1] * R_d - x[2 * n]
    )

    # Magnetic flux conservation at each node
    f[n] = -x[n] + Fi_ld + x[0]
    for i in range(1, n):
        f[n + i] = -x[n + i] + x[n + i - 1] + x[i]

    return f


def solve_static_mec_point(delta, Fi_delta):
    """Solve one static MEC operating point via Newton-Raphson and adaptive slicing."""
    Fi_ld = Fi_delta
    S_cd = np.pi * (r3**2 - r2**2)
    S_cr = np.pi * r1**2
    S_ld = 0.25 * (r3 + r2 - r1) * np.pi * b_ld
    S_bs = 0.25 * (r3 + r2 - r1) * np.pi * b_bs
    S_delta1, S_delta2 = S_cr, S_cd
    S_ldv1, S_ldv2 = S_delta1, S_delta2
    S_bsv1, S_bsv2 = S_cr, S_cd

    l_s_cr = l_cr
    l_s_ld = 0.5 * (r3 + r2 - r1)
    l_s_bs = l_s_ld
    l_s_ldv = 0.5 * b_ld
    l_s_bsv = 0.5 * b_bs

    R_delta1 = delta / (mu_0 * S_delta1)
    R_delta2 = delta / (mu_0 * S_delta2)
    lyambda_d = mu_0 * 2.0 * np.pi / np.log(r2 / r1)

    rng = np.random.default_rng(42)
    n = 1
    while n < 5000:
        l_s_crk = l_s_cr / (n + 1)
        R_d = 1.0 / (l_s_crk * lyambda_d)
        dim = 2 * n + 1
        x = 0.0001 * rng.random(dim)
        tol = 1e-8
        dx = 1e-6

        for _ in range(100):
            f = compute_residuals(
                x, n, R_d, Fi_ld, S_ld, S_ldv1, S_ldv2, S_bsv1, S_bsv2,
                S_cr, S_cd, S_bs, l_s_crk, l_s_ld, l_s_ldv, l_s_bs, l_s_bsv,
                R_delta1, R_delta2
            )
            if np.linalg.norm(f) < tol:
                break

            J = np.zeros((dim, dim))
            for j in range(dim):
                x1 = x.copy()
                x1[j] += dx
                f1 = compute_residuals(
                    x1, n, R_d, Fi_ld, S_ld, S_ldv1, S_ldv2, S_bsv1, S_bsv2,
                    S_cr, S_cd, S_bs, l_s_crk, l_s_ld, l_s_ldv, l_s_bs, l_s_bsv,
                    R_delta1, R_delta2
                )
                J[:, j] = (f1 - f) / dx

            x = x + np.linalg.solve(J, -f)

        if n > 1 and (abs(x[2 * n - 1] - x[2 * n - 2]) < 0.01 * abs(x[2 * n - 1])):
            break
        n += 1

    total_mmf = x[2 * n] * (n + 1)
    magnetic_force = (Fi_ld**2 / (2.0 * mu_0)) * (1.0 / S_delta1 + 1.0 / S_delta2)
    avg_core_flux = (Fi_ld + np.sum(x[n : 2 * n])) / (n + 1)
    return total_mmf, magnetic_force, avg_core_flux


# 3. Sweep (delta, current) grid and build rectangular tables
delta_grid = np.linspace(min_air_gap, max_air_gap, 30)
current_grid = np.linspace(0.0, max_current, 60)
Fmag_table = np.zeros((len(delta_grid), len(current_grid)))
psi_table = np.zeros((len(delta_grid), len(current_grid)))

print("Generating rectangular MEC surrogate lookup tables (takes ~20 seconds)...")
for p_idx, delta in enumerate(delta_grid):
    I_row, F_row, Psi_row = [0.0], [0.0], [0.0]
    Fi_delta = 0.02e-4
    while I_row[-1] < max_current * 1.05:
        F_mmf, F_mag, Phi_avg = solve_static_mec_point(delta, Fi_delta)
        I_row.append(F_mmf / coil_turns)
        F_row.append(F_mag)
        Psi_row.append(coil_turns * Phi_avg)
        Fi_delta += 0.08e-4

    I_arr, F_arr, Psi_arr = np.array(I_row), np.array(F_row), np.array(Psi_row)
    _, unique_idx = np.unique(I_arr, return_index=True)
    unique_idx = np.sort(unique_idx)

    Fmag_table[p_idx, :] = interp1d(
        I_arr[unique_idx], F_arr[unique_idx], kind="cubic", fill_value="extrapolate"
    )(current_grid)
    psi_table[p_idx, :] = interp1d(
        I_arr[unique_idx], Psi_arr[unique_idx], kind="cubic", fill_value="extrapolate"
    )(current_grid)

# Build inverse table i(delta, Psi) for the electrical ODE
max_common_psi = float(np.min(psi_table[:, -1]))
psi_grid = np.linspace(0.0, max_common_psi, 60)
current_from_psi_table = np.zeros((len(delta_grid), len(psi_grid)))

for p_idx in range(len(delta_grid)):
    Psi_row = psi_table[p_idx, :]
    _, unique_idx = np.unique(Psi_row, return_index=True)
    unique_idx = np.sort(unique_idx)
    current_from_psi_table[p_idx, :] = interp1d(
        Psi_row[unique_idx], current_grid[unique_idx], kind="cubic", fill_value="extrapolate"
    )(psi_grid)

# 4. Save to .mat file in the current directory
sio.savemat("mec_rectangular_surrogate.mat", {
    "delta_grid": delta_grid,
    "current_grid": current_grid,
    "psi_grid": psi_grid,
    "Fmag_table": Fmag_table,
    "psi_table": psi_table,
    "current_from_psi_table": current_from_psi_table,
    "coil_turns": coil_turns,
    "coil_resistance": coil_resistance,
    "load_mass": load_mass,
    "max_voltage": max_voltage,
    "max_current": max_current,
    "min_air_gap": min_air_gap,
    "max_air_gap": max_air_gap,
})

print(f"Saved 'mec_rectangular_surrogate.mat' successfully in {time.time() - start_time:.1f} seconds!")
