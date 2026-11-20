import numpy as np
import scipy.io as sio
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

# 1. Load precomputed rectangular MEC surrogate tables
mat = sio.loadmat("mec_rectangular_surrogate.mat", squeeze_me=True)
delta_grid = mat["delta_grid"]
current_grid = mat["current_grid"]
psi_grid = mat["psi_grid"]
Fmag_table = mat["Fmag_table"]
psi_table = mat["psi_table"]
i_from_psi_table = mat["current_from_psi_table"]

R_coil = float(mat["coil_resistance"])
mass = float(mat["load_mass"])
V_max = float(mat["max_voltage"])
d_min = float(mat["min_air_gap"])
d_max = float(mat["max_air_gap"])
g = 9.81

# =========================================================================
# FIGURE 1: 3D NONLINEAR SURROGATE SURFACES F_mag(delta, i) AND Psi(delta, i)
# =========================================================================
D_mesh, I_mesh = np.meshgrid(delta_grid * 1e3, current_grid, indexing="ij")

fig1 = plt.figure(figsize=(12, 5))
fig1.canvas.manager.set_window_title("Figure 1: Nonlinear MEC 3D Surrogate Surfaces")

ax1 = fig1.add_subplot(1, 2, 1, projection="3d")
ax1.plot_surface(D_mesh, I_mesh, Fmag_table, cmap="viridis", edgecolor="none", alpha=0.92)
ax1.set_xlabel("Air Gap δ (mm)")
ax1.set_ylabel("Coil Current i (A)")
ax1.set_zlabel("Magnetic Force F_mag (N)")
ax1.set_title("Magnetic Pulling Force F_mag(δ, i)")

ax2 = fig1.add_subplot(1, 2, 2, projection="3d")
ax2.plot_surface(D_mesh, I_mesh, psi_table, cmap="plasma", edgecolor="none", alpha=0.92)
ax2.set_xlabel("Air Gap δ (mm)")
ax2.set_ylabel("Coil Current i (A)")
ax2.set_zlabel("Flux Linkage Ψ (Wb-turns)")
ax2.set_title("Coil Flux Linkage Ψ(δ, i)")
plt.tight_layout()

# =========================================================================
# FIGURE 2: 4-PANEL COUPLED DYNAMIC STABILIZATION RESPONSE (1.1 mm -> 0.6 mm)
# =========================================================================
interp_F = RegularGridInterpolator((delta_grid, current_grid), Fmag_table, bounds_error=False, fill_value=None)
interp_Psi = RegularGridInterpolator((delta_grid, current_grid), psi_table, bounds_error=False, fill_value=None)
interp_I = RegularGridInterpolator((delta_grid, psi_grid), i_from_psi_table, bounds_error=False, fill_value=None)

delta_ref = 0.6e-3
f_curve = np.array([float(interp_F([[delta_ref, iv]])[0]) for iv in current_grid])
i_eq = float(np.interp(mass * g, f_curve, current_grid))
psi_eq = float(interp_Psi([[delta_ref, i_eq]])[0])
u_eq = R_coil * i_eq

dt = 1.0 / 2000.0
steps = int(0.4 / dt)
state = np.array([1.1e-3, 0.0, psi_eq])  # Start at 1.1 mm
Kp, Kd, K_psi = 10000.0, 200.0, 250.0

t_arr, d_arr, v_arr, i_arr, u_arr = [], [], [], [], []

def deriv(s, u):
    dc = np.clip(s[0], delta_grid[0], delta_grid[-1])
    pc = np.clip(s[2], psi_grid[0], psi_grid[-1])
    iv = float(interp_I([[dc, pc]])[0])
    fv = float(interp_F([[dc, np.clip(iv, 0, current_grid[-1])]])[0])
    dv = g - fv / mass
    if (s[0] <= d_min and dv < 0) or (s[0] >= d_max and dv > 0):
        dv = 0.0
    return np.array([s[1], dv, u - R_coil * iv]), iv

for step in range(steps):
    t = step * dt
    u = float(np.clip(u_eq + Kp * (state[0] - delta_ref) + Kd * state[1] - K_psi * (state[2] - psi_eq), -V_max, V_max))
    k1, curr = deriv(state, u)
    k2, _ = deriv(state + 0.5 * dt * k1, u)
    k3, _ = deriv(state + 0.5 * dt * k2, u)
    k4, _ = deriv(state + dt * k3, u)
    state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    state[0] = np.clip(state[0], d_min, d_max)
    state[2] = np.clip(state[2], psi_grid[0], psi_grid[-1])

    t_arr.append(t * 1e3)
    d_arr.append(state[0] * 1e3)
    v_arr.append(state[1] * 1e3)
    i_arr.append(curr)
    u_arr.append(u)

fig2, axs = plt.subplots(2, 2, figsize=(11, 6))
fig2.canvas.manager.set_window_title("Figure 2: Coupled Dynamic Stabilization (PD Baseline)")

axs[0, 0].plot(t_arr, d_arr, "b-", lw=2)
axs[0, 0].axhline(0.6, color="r", ls="--", lw=1.5, label="Target δ_ref = 0.6 mm")
axs[0, 0].set_title("Air-Gap Stabilization")
axs[0, 0].set_xlabel("Time (ms)")
axs[0, 0].set_ylabel("Air Gap δ (mm)")
axs[0, 0].legend()
axs[0, 0].grid(True)

axs[0, 1].plot(t_arr, v_arr, "k-", lw=2)
axs[0, 1].set_title("Load Vertical Velocity")
axs[0, 1].set_xlabel("Time (ms)")
axs[0, 1].set_ylabel("Velocity v (mm/s)")
axs[0, 1].grid(True)

axs[1, 0].plot(t_arr, i_arr, "m-", lw=2)
axs[1, 0].set_title("Electrical Coil Current")
axs[1, 0].set_xlabel("Time (ms)")
axs[1, 0].set_ylabel("Current i (A)")
axs[1, 0].grid(True)

axs[1, 1].plot(t_arr, u_arr, "g-", lw=2)
axs[1, 1].set_title("Applied Control Voltage (Note Initial Ringing)")
axs[1, 1].set_xlabel("Time (ms)")
axs[1, 1].set_ylabel("Voltage u (V)")
axs[1, 1].grid(True)

plt.tight_layout()

fig1.savefig("fig1_3d_surfaces.png", dpi=300)
fig2.savefig("fig2_dynamic_response.png", dpi=300)
plt.show()
