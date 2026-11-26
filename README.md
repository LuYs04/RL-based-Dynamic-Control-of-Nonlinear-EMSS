# Reinforcement Learning-Based Dynamic Control and Stabilization of Nonlinear Electromagnetic Suspension Systems

Master's thesis project focused on the coupled electromechanical modeling, 3D physics simulation, and deep reinforcement learning control (**TD3**) of an inherently unstable nonlinear **Electromagnetic Suspension (EMS)** system.

## Overview

Electromagnetic suspension systems levitate a ferromagnetic load without mechanical contact using magnetic attraction. Because magnetic force scales nonlinearly with the air gap and ferromagnetic core saturation ($B\text{–}H$ curve), open-loop levitation is inherently unstable and requires fast, active feedback control.

This repository bridges high-fidelity electromagnetic modeling in **MATLAB** with real-time 3D rigid-body simulation in **Python / PyBullet** to train and benchmark a **Twin Delayed Deep Deterministic Policy Gradient (TD3)** controller against classical baseline controllers.

## Key Features & Architecture

1. **Nonlinear Magnetic Equivalent Circuit (MEC) Modeling (MATLAB & Python)**
   - Models a cylindrical pot-core electromagnet accounting for ferromagnetic saturation ($B\text{–}H$ curve), coaxial radial flux leakage via adaptive axial discretization, and Maxwell's stress tensor for magnetic pulling force computation.
   - Extends the static MEC formulation with coupled electromechanical parameters to capture transient system behavior.
2. **Fast Surrogate Lookup Tables**
   - Precomputes 2D rectangular surrogate grids (`mec_rectangular_surrogate.mat`) for rapid bicubic/bilinear interpolation during real-time simulation and RL training, avoiding expensive iterative Newton-Raphson solves at every step.
3. **Interactive 3D PyBullet Simulations**
   - **`StaticMEC.py`**: Quasi-static simulation coupling the nonlinear MEC force map with 3D rigid-body mechanics.
   - **`DynamicMEC.py`**: Full coupled electromechanical simulation integrating mechanical kinematics/dynamics and Faraday's law of induction via 4th-order Runge-Kutta (RK4), complete with interactive sliders for target air gap, load mass, and external force disturbances.
4. **Reinforcement Learning Control (Gymnasium + TD3)** *(In Progress)*
   - Custom `gymnasium.Env` wrapper around the coupled nonlinear EMS dynamics to train a continuous-action TD3 agent for robust air-gap stabilization and disturbance rejection.

## Repository Structure

```text
├── matlab/                     # Static & coupled dynamic MEC solvers (ode45)
├── generate_mec_surrogate.py   # Generates rectangular 2D MEC lookup tables (.mat)
├── StaticMEC.py                # Quasi-static 3D PyBullet visualization
├── DynamicMEC.py               # Full coupled dynamic 3D PyBullet simulation (RK4)
└── README.md
