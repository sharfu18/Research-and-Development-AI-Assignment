# -*- coding: utf-8 -*-
"""
Author: Shaik Sharfuddin
Title: Parameter Estimation for R&D (AI) Assignment
Description: Estimates θ, M, X for a given parametric curve
             using nonlinear least-squares optimization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

# Path to dataset
DATA_FILE = r"C:\Users\sharf\OneDrive - Amrita Vishwa Vidyapeetham\Desktop\xy_data.csv"

# Range of parameter t
T_START, T_END = 6, 60

# Allowed limits for parameters
THETA_LIMITS = (np.deg2rad(0), np.deg2rad(50))     # radians
M_LIMITS = (-0.05, 0.05)
X_LIMITS = (0.0, 100.0)

# ---------------------------------------------------------------------------
# MODEL EQUATIONS
# ---------------------------------------------------------------------------

def curve_model(t_vals, theta, M, X):
    """
    Computes the parametric curve coordinates (x, y)
    for given parameter values (theta, M, X).
    """
    t_vals = np.asarray(t_vals)
    exp_term = np.exp(M * np.abs(t_vals)) * np.sin(0.3 * t_vals)

    x_vals = t_vals * np.cos(theta) - exp_term * np.sin(theta) + X
    y_vals = 42 + t_vals * np.sin(theta) + exp_term * np.cos(theta)
    return x_vals, y_vals


def residual_vector(params, t_vals, x_real, y_real):
    """
    Returns the residuals between observed (x, y)
    and model-predicted (x, y) for least squares fitting.
    """
    theta, M, X = params
    x_pred, y_pred = curve_model(t_vals, theta, M, X)
    return np.concatenate([(x_pred - x_real), (y_pred - y_real)])


# ---------------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------------

data = pd.read_csv(DATA_FILE)
data.columns = [col.lower() for col in data.columns]

# If t is not given, assign equally spaced t values
if 't' in data.columns:
    t_data = data['t'].values
else:
    t_data = np.linspace(T_START, T_END, len(data))

x_data = data['x'].values
y_data = data['y'].values

# ---------------------------------------------------------------------------
# INITIAL ESTIMATES AND OPTIMIZATION
# ---------------------------------------------------------------------------

initial_guess = [np.deg2rad(25), 0.0, 50.0]

param_bounds = (
    [THETA_LIMITS[0], M_LIMITS[0], X_LIMITS[0]],
    [THETA_LIMITS[1], M_LIMITS[1], X_LIMITS[1]]
)

# Perform least-squares optimization
fit_result = least_squares(
    residual_vector,
    x0=initial_guess,
    bounds=param_bounds,
    args=(t_data, x_data, y_data),
    verbose=2
)

theta_opt, M_opt, X_opt = fit_result.x

# ---------------------------------------------------------------------------
# RESULTS
# ---------------------------------------------------------------------------

print("\n--- Estimated Parameters ---")
print(f"Theta  : {np.degrees(theta_opt):.4f} degrees")
print(f"M      : {M_opt:.6f}")
print(f"X      : {X_opt:.6f}")

# ---------------------------------------------------------------------------
# L1 METRIC (Assignment Score)
# ---------------------------------------------------------------------------

def compute_L1(theta, M, X):
    t_uniform = np.linspace(T_START, T_END, 500)
    x_model, y_model = curve_model(t_uniform, theta, M, X)

    x_ref = np.interp(t_uniform, np.linspace(T_START, T_END, len(x_data)), x_data)
    y_ref = np.interp(t_uniform, np.linspace(T_START, T_END, len(y_data)), y_data)

    return np.mean(np.abs(x_model - x_ref) + np.abs(y_model - y_ref))

L1_error = compute_L1(theta_opt, M_opt, X_opt)
print(f"\nL1 Distance (Evaluation Metric): {L1_error:.6f}")

# ---------------------------------------------------------------------------
# DESMOS EQUATION STRING
# ---------------------------------------------------------------------------

desmos_expr = (
    rf"\left(t*\cos({theta_opt:.6f}) - e^{{{M_opt:.6f}\left|t\right|}}\sin(0.3t)\sin({theta_opt:.6f}) + {X_opt:.6f}, "
    rf"42 + t*\sin({theta_opt:.6f}) + e^{{{M_opt:.6f}\left|t\right|}}\sin(0.3t)\cos({theta_opt:.6f})\right)"
)

print("\n--- Copy for Desmos Submission ---")
print(desmos_expr)

# ---------------------------------------------------------------------------
# VISUALIZATION
# ---------------------------------------------------------------------------

x_fit, y_fit = curve_model(t_data, theta_opt, M_opt, X_opt)

plt.figure(figsize=(7, 5))
plt.scatter(x_data, y_data, color='red', s=15, label='Observed Data')
plt.plot(x_fit, y_fit, color='blue', linewidth=1.5, label='Optimized Fit')
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.title("Parametric Curve Fitting (R&D Assignment)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
