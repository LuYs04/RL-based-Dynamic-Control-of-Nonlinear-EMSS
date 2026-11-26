import time
import numpy as np
import scipy.io as sio
from scipy.interpolate import RegularGridInterpolator
import pybullet as p
import pybullet_data

# =========================================================================
# INTERACTIVE FULL DYNAMIC MEC SIMULATION IN PYBULLET (VOLTAGE-CONTROLLED)
# =========================================================================
# Integrates the full 3-state coupled nonlinear electromechanical ODEs via
# 4th-order Runge-Kutta (RK4):
#   1. Kinematics:  d(delta)/dt = v
#   2. Mechanics:   m * dv/dt   = m * g + f_disturb - F_mag(delta, i)
#   3. Electrics:   d(Psi)/dt   = u - R_coil * i(delta, Psi) [Faraday's Law]
# =========================================================================

# 1. Load precomputed rectangular MEC surrogate tables
mat_data = sio.loadmat("mec_rectangular_surrogate.mat", squeeze_me=True)

delta_grid = mat_data["delta_grid"]                         # 1D air-gap grid [m]
current_grid = mat_data["current_grid"]                     # 1D coil current grid [A]
psi_grid = mat_data["psi_grid"]                             # 1D flux linkage grid [Wb-turns]
Fmag_table = mat_data["Fmag_table"]                         # 2D force table F_mag(delta, i) [N]
psi_table = mat_data["psi_table"]                           # 2D flux linkage table Psi(delta, i) [Wb-t]
current_from_psi_table = mat_data["current_from_psi_table"] # 2D inverse table i(delta, Psi) [A]

R_coil = float(mat_data["coil_resistance"])                 # Coil resistance [Ohm]
default_mass = float(mat_data["load_mass"])                 # Default load mass [kg]
V_max = float(mat_data["max_voltage"])                      # Maximum supply voltage [V]
delta_min = float(mat_data["min_air_gap"])                  # Min mechanical stop [m]
delta_max = float(mat_data["max_air_gap"])                  # Max mechanical stop [m]
g = 9.81                                                    # Gravity [m/s^2]

interp_Fmag = RegularGridInterpolator(
    (delta_grid, current_grid),
    Fmag_table,
    bounds_error=False,
    fill_value=None,
)
interp_Psi = RegularGridInterpolator(
    (delta_grid, current_grid),
    psi_table,
    bounds_error=False,
    fill_value=None,
)
interp_I_from_Psi = RegularGridInterpolator(
    (delta_grid, psi_grid),
    current_from_psi_table,
    bounds_error=False,
    fill_value=None,
)

def compute_equilibrium(target_gap, mass):
    """Compute steady-state (i_eq, psi_eq, u_eq) balancing gravity at target_gap."""
    d_c = float(np.clip(target_gap, delta_grid[0], delta_grid[-1]))
    f_curve = np.array([float(interp_Fmag([[d_c, iv]])[0]) for iv in current_grid])
    i_eq = float(np.interp(mass * g, f_curve, current_grid))
    psi_eq = float(interp_Psi([[d_c, i_eq]])[0])
    u_eq = R_coil * i_eq
    return i_eq, psi_eq, u_eq

# 2. Initialize PyBullet 3D GUI (clean layout with interactive Params panel)
r1, r2, r3 = 10.45e-3, 25.40e-3, 27.55e-3
b_ld, b_bs, l_cr = 6.30e-3, 7.30e-3, 29.50e-3
z_pole_face = 0.050

physics_client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, 0)  # Exact 3-state coupled ODEs integrated via RK4 below

# Hide the 3 unused camera preview boxes on the left; keep the Params sliders on the right
p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)

p.resetDebugVisualizerCamera(
    cameraDistance=0.11,
    cameraYaw=35,
    cameraPitch=-12,
    cameraTargetPosition=[0, 0, z_pole_face - 0.002],
)

# Build top-mounted stationary pot-core electromagnet
vis_base = p.createVisualShape(p.GEOM_CYLINDER, radius=r3, length=b_bs, rgbaColor=[0.25, 0.28, 0.32, 1.0])
vis_core = p.createVisualShape(p.GEOM_CYLINDER, radius=r1, length=l_cr, rgbaColor=[0.35, 0.38, 0.42, 1.0])
vis_coil = p.createVisualShape(p.GEOM_CYLINDER, radius=r2 * 0.96, length=l_cr * 0.85, rgbaColor=[0.80, 0.45, 0.20, 0.85])
vis_shell = p.createVisualShape(p.GEOM_CYLINDER, radius=r3, length=l_cr, rgbaColor=[0.25, 0.28, 0.32, 0.30])

stator_id = p.createMultiBody(
    baseMass=0,
    baseVisualShapeIndex=vis_base,
    basePosition=[0, 0, z_pole_face + l_cr + b_bs / 2.0],
    linkMasses=[0, 0, 0],
    linkCollisionShapeIndices=[-1, -1, -1],
    linkVisualShapeIndices=[vis_core, vis_coil, vis_shell],
    linkPositions=[
        [0, 0, -(b_bs / 2.0 + l_cr / 2.0)],
        [0, 0, -(b_bs / 2.0 + l_cr / 2.0)],
        [0, 0, -(b_bs / 2.0 + l_cr / 2.0)],
    ],
    linkOrientations=[[0, 0, 0, 1]] * 3,
    linkInertialFramePositions=[[0, 0, 0]] * 3,
    linkInertialFrameOrientations=[[0, 0, 0, 1]] * 3,
    linkParentIndices=[0, 0, 0],
    linkJointTypes=[p.JOINT_FIXED] * 3,
    linkJointAxis=[[0, 0, 1]] * 3,
)

# Build movable suspended load disk below the electromagnet
delta_init = 1.5e-3  # Start at 1.50 mm so upward motion is clearly visible
vis_load = p.createVisualShape(p.GEOM_CYLINDER, radius=r3, length=b_ld, rgbaColor=[0.15, 0.55, 0.85, 0.95])
load_id = p.createMultiBody(
    baseMass=0,
    baseVisualShapeIndex=vis_load,
    basePosition=[0, 0, z_pole_face - delta_init - b_ld / 2.0],
)

# 3. Add Interactive Sliders in the Right-Hand "Params" Panel
slider_target_gap = p.addUserDebugParameter("Target Air Gap (mm)", 0.20, 1.80, 0.60)
slider_disturbance = p.addUserDebugParameter("Push / Pull Force (N)", -8.0, 8.0, 0.0)
slider_mass = p.addUserDebugParameter("Load Mass (kg)", 0.20, 1.50, default_mass)
btn_reset = p.addUserDebugParameter("Reset Drop (Click)", 1, 0, 0)

# 4. Coupled Electromechanical RK4 Stepper
Kp_u = 35000.0  # Proportional voltage gain [V/m]
Kd_u = 900.0    # Derivative voltage gain [V/(m/s)]

def ode_derivatives(state, voltage, mass, f_disturb):
    d_val, v_val, psi_val = state
    d_c = float(np.clip(d_val, delta_grid[0], delta_grid[-1]))
    p_c = float(np.clip(psi_val, psi_grid[0], psi_grid[-1]))
    i_val = float(interp_I_from_Psi([[d_c, p_c]])[0])
    i_c = float(np.clip(i_val, current_grid[0], current_grid[-1]))
    f_val = float(interp_Fmag([[d_c, i_c]])[0])

    dd_dt = v_val
    dv_dt = g + (f_disturb - f_val) / mass
    if (d_val <= delta_min and dv_dt < 0.0) or (d_val >= delta_max and dv_dt > 0.0):
        dv_dt = 0.0
    dpsi_dt = voltage - R_coil * i_val
    return np.array([dd_dt, dv_dt, dpsi_dt]), i_val, f_val

# Initialize state [delta, vel, psi]
_, psi_init, _ = compute_equilibrium(0.60e-3, default_mass)
state = np.array([delta_init, 0.0, psi_init])
dt = 1.0 / 2000.0  # 0.5 ms RK4 integration step

force_arrow_id = -1
target_line_id = -1
text_id_1 = -1
text_id_2 = -1
last_reset_val = p.readUserDebugParameter(btn_reset)
step = 0

try:
    while p.isConnected():
        # Read interactive sliders
        delta_ref = p.readUserDebugParameter(slider_target_gap) * 1e-3
        f_disturb = p.readUserDebugParameter(slider_disturbance)
        load_mass = p.readUserDebugParameter(slider_mass)
        reset_val = p.readUserDebugParameter(btn_reset)

        if reset_val != last_reset_val:
            _, psi_rst, _ = compute_equilibrium(delta_ref, load_mass)
            state = np.array([1.5e-3, 0.0, psi_rst])
            last_reset_val = reset_val

        # Compute feedforward equilibrium voltage & feedback control law (later TD3 RL policy)
        _, _, u_eq = compute_equilibrium(delta_ref, load_mass)
        voltage = float(np.clip(u_eq + Kp_u * (state[0] - delta_ref) + Kd_u * state[1], -V_max, V_max))

        # 4th-Order Runge-Kutta (RK4) step for coupled electromechanical ODEs
        k1, current, f_mag = ode_derivatives(state, voltage, load_mass, f_disturb)
        k2, _, _ = ode_derivatives(state + 0.5 * dt * k1, voltage, load_mass, f_disturb)
        k3, _, _ = ode_derivatives(state + 0.5 * dt * k2, voltage, load_mass, f_disturb)
        k4, _, _ = ode_derivatives(state + dt * k3, voltage, load_mass, f_disturb)
        state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # Enforce hard mechanical & flux bounds
        if state[0] <= delta_min:
            state[0] = delta_min
            if state[1] < 0.0:
                state[1] = 0.0
        elif state[0] >= delta_max:
            state[0] = delta_max
            if state[1] > 0.0:
                state[1] = 0.0
        state[2] = float(np.clip(state[2], psi_grid[0], psi_grid[-1]))

        delta = state[0]

        # Update 3D position of the suspended load in PyBullet
        z_load_center = z_pole_face - delta - b_ld / 2.0
        p.resetBasePositionAndOrientation(load_id, [0, 0, z_load_center], [0, 0, 0, 1])

        # Render visuals every 40 physics steps (20 ms -> 50 FPS)
        if step % 10 == 0:
            p.stepSimulation()

            # Red reference line showing exact target air gap height
            z_target_top = z_pole_face - delta_ref
            target_line_id = p.addUserDebugLine(
                lineFromXYZ=[-r3 * 1.25, 0, z_target_top],
                lineToXYZ=[r3 * 1.25, 0, z_target_top],
                lineColorRGB=[0.95, 0.15, 0.15],
                lineWidth=3,
                replaceItemUniqueId=target_line_id,
            )

            # Green arrow showing upward magnetic pulling force F_mag
            arrow_len = min(f_mag * 0.0025, 0.025)
            force_arrow_id = p.addUserDebugLine(
                lineFromXYZ=[0, 0, z_load_center - b_ld / 2.0 - arrow_len],
                lineToXYZ=[0, 0, z_load_center - b_ld / 2.0],
                lineColorRGB=[0.1, 0.8, 0.2],
                lineWidth=4,
                replaceItemUniqueId=force_arrow_id,
            )

            # Two-line HUD telemetry centered above the magnet
            text_id_1 = p.addUserDebugText(
                text=f"[DYNAMIC MEC - Voltage + Inductance] Gap: {delta*1e3:.3f} mm | Target: {delta_ref*1e3:.2f} mm",
                textPosition=[-0.032, 0, z_pole_face + l_cr + 0.017],
                textColorRGB=[0.75, 0.15, 0.05],
                textSize=1.1,
                replaceItemUniqueId=text_id_1,
            )
            text_id_2 = p.addUserDebugText(
                text=f"Voltage u: {voltage:.2f} V | Current i: {current:.2f} A | F_mag: {f_mag:.2f} N",
                textPosition=[-0.032, 0, z_pole_face + l_cr + 0.012],
                textColorRGB=[0.1, 0.1, 0.1],
                textSize=1.05,
                replaceItemUniqueId=text_id_2,
            )
            time.sleep(0.1)

        step += 1

except KeyboardInterrupt:
    pass
finally:
    if p.isConnected():
        p.disconnect()
