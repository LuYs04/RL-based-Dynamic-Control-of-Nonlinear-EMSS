import time
import numpy as np
import scipy.io as sio
from scipy.interpolate import RegularGridInterpolator
import pybullet as p
import pybullet_data

# =========================================================================
# INTERACTIVE QUASI-STATIC MEC SIMULATION IN PYBULLET (CURRENT-CONTROLLED)
# =========================================================================
# Electrical coil inductance (dPsi/dt) is neglected: current i(t) responds
# instantaneously to controller commands, and F_mag(delta, i) updates
# immediately from the precomputed static MEC lookup table.
# =========================================================================

# 1. Load precomputed rectangular MEC surrogate tables
mat_data = sio.loadmat("mec_rectangular_surrogate.mat", squeeze_me=True)

delta_grid = mat_data["delta_grid"]          # 1D air-gap grid [m]
current_grid = mat_data["current_grid"]      # 1D coil current grid [A]
Fmag_table = mat_data["Fmag_table"]          # 2D magnetic force table F_mag(delta, i) [N]

R_coil = float(mat_data["coil_resistance"])  # Coil resistance [Ohm]
default_mass = float(mat_data["load_mass"])  # Default load mass [kg]
I_max = float(mat_data["max_current"])       # Maximum coil current [A]
delta_min = float(mat_data["min_air_gap"])   # Min mechanical stop [m]
delta_max = float(mat_data["max_air_gap"])   # Max mechanical stop [m]
g = 9.81                                     # Gravity [m/s^2]

interp_Fmag = RegularGridInterpolator(
    (delta_grid, current_grid),
    Fmag_table,
    bounds_error=False,
    fill_value=None,
)

def compute_eq_current(target_gap, mass):
    """Find steady-state equilibrium current i_eq such that F_mag(gap, i_eq) = m * g."""
    d_c = float(np.clip(target_gap, delta_grid[0], delta_grid[-1]))
    f_curve = np.array([float(interp_Fmag([[d_c, iv]])[0]) for iv in current_grid])
    return float(np.interp(mass * g, f_curve, current_grid))

# 2. Initialize PyBullet 3D GUI (clean layout with interactive Params panel)
r1, r2, r3 = 10.45e-3, 25.40e-3, 27.55e-3
b_ld, b_bs, l_cr = 6.30e-3, 7.30e-3, 29.50e-3
z_pole_face = 0.050

physics_client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, 0)  # Exact 1D vertical dynamics integrated explicitly below

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

# 4. Real-Time Quasi-Static Simulation Loop
dt = 1.0 / 2000.0  # 0.5 ms physics step
delta = delta_init
vel = 0.0          # Downward velocity d(delta)/dt [m/s]

Kp_i = 35000.0 / R_coil
Kd_i = 900.0 / R_coil

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
            delta = 1.5e-3
            vel = 0.0
            last_reset_val = reset_val

        # Compute instantaneous equilibrium current for current target gap & mass
        i_eq = compute_eq_current(delta_ref, load_mass)

        # Direct current control law (zero electrical inductance lag)
        current = float(np.clip(i_eq + Kp_i * (delta - delta_ref) + Kd_i * vel, 0.0, I_max))

        # Look up upward magnetic pulling force F_mag(delta, i)
        d_clamped = float(np.clip(delta, delta_grid[0], delta_grid[-1]))
        f_mag = float(interp_Fmag([[d_clamped, current]])[0])

        # Integrate 1D vertical mechanics: m * dv/dt = m * g + f_disturb - F_mag
        acc = g + (f_disturb - f_mag) / load_mass
        vel += acc * dt
        delta += vel * dt

        # Enforce hard mechanical stops [delta_min, delta_max]
        if delta <= delta_min:
            delta = delta_min
            if vel < 0.0:
                vel = 0.0
        elif delta >= delta_max:
            delta = delta_max
            if vel > 0.0:
                vel = 0.0

        # Update 3D position of the suspended load in PyBullet
        z_load_center = z_pole_face - delta - b_ld / 2.0
        p.resetBasePositionAndOrientation(load_id, [0, 0, z_load_center], [0, 0, 0, 1])

        # Render visuals every 40 physics steps (20 ms -> 50 FPS)
        if step % 40 == 0:
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
                text=f"[STATIC MEC - Instant Current] Gap: {delta*1e3:.3f} mm | Target: {delta_ref*1e3:.2f} mm",
                textPosition=[-0.032, 0, z_pole_face + l_cr + 0.017],
                textColorRGB=[0.0, 0.35, 0.75],
                textSize=1.1,
                replaceItemUniqueId=text_id_1,
            )
            text_id_2 = p.addUserDebugText(
                text=f"Current i: {current:.2f} A | F_mag: {f_mag:.2f} N | Gravity mg: {load_mass*g:.2f} N",
                textPosition=[-0.032, 0, z_pole_face + l_cr + 0.012],
                textColorRGB=[0.1, 0.1, 0.1],
                textSize=1.05,
                replaceItemUniqueId=text_id_2,
            )
            time.sleep(0.015)

        step += 1

except KeyboardInterrupt:
    pass
finally:
    if p.isConnected():
        p.disconnect()
