"""

    Equivariance based CBF synthesis for the bicycle model using the knowledge over a partially known CBF. 
    The constraint under consideration is an

    >> elliptical obstacle

    Adrian Wiltz, 2025

"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import numpy as np
from Dynamics.Bicycle import Bicycle
from CBF.CBFmodule import CBFmodule
import CBF.CBFcomputation as CBFcomputation
from Equivariance_Module import equi_cbf
from math_aux.math_aux import wrap_to_pi
from scipy.optimize import root_scalar
import casadi as ca
import time

########################################################################################

cbf_module_filename = "2025-06-17_16-41-39_cbf_straight_line_partial.json"

cbf_module_folder_path = r'Examples_without_symmetry/Data_cbf_partial'

cbf_complete_file_name = 'cbf_ellipse.json'

########################################################################################
# load partially known cbf

cbfModule_partial = CBFmodule()
cbfModule_partial.load(cbf_module_filename, cbf_module_folder_path)

########################################################################################
# Constraint specifications

a = 5
b = 3

def h(x):
    """
    Constraint function h for the circle with radius 4.
    
    Args:
        x (array-like): The point in the domain.
        
    Returns:
        h_value (float): The value of the constraint function.
    
    """
    
    a = 5
    b = 3
    
    return (x[0]**2 / a**2) + (x[1]**2 / b**2) - 1

########################################################################################
# Some CBF settings

domain_lower_bound_complete = np.array([-10,-10,-np.pi])
domain_upper_bound_complete = np.array([10,10,np.pi])
discretization_complete = np.array([21,21,41])

########################################################################################
# Define transformation and parameter function

def sigma_func(sigma, x, y, a=a, b=b):
    """
    Computes the residual of the implicit equation involving sigma (σ), derived from the
    parametric relationship between a point (x, y) and a point on an ellipse defined by (a, b).

    This function represents the difference between the left and right sides of the equation:

        y * cos(σ) = b * sin(σ) * cos(σ) + a * (x - a * cos(σ)/b) * sin(σ)

    Rearranged into a root-finding form:
        sigma_func(σ) = b * sin(σ) * cos(σ) + a * (x - a * cos(σ)/b) * sin(σ) - y  * cos(σ) = 0

    A zero of this function corresponds to a value of σ (in radians) for which the original
    equation holds true — i.e., the point (x, y) lies on or corresponds to that position
    on the ellipse parameterized by σ.

    Parameters:
    -----------
    sigma : float or np.ndarray
        The angle in radians. Can be a single float or an array of floats.
    x : float
        x-coordinate of the point from which the relationship is computed.
    y : float
        y-coordinate of the point from which the relationship is computed.
    a : float
        Semi-major axis of the ellipse.
    b : float
        Semi-minor axis of the ellipse.

    Returns:
    --------
    float or np.ndarray
        The value(s) of the function f(σ). A root of this function indicates a solution σ
        for which the parametric ellipse expression passes through or aligns with (x, y).

    Notes:
    ------
    - This function may be used with root-finding methods (e.g., Brent's method) to solve
      for σ over a given interval.
    - If sigma is an array, the result is a NumPy array of residuals for each input.
    - Avoid divisions by zero: cos(σ) in the denominator must not be zero.
    """

    # regularized implementation

    term1 = b * 0.5 * np.sin(2*sigma)     # use the double angle identity for sin(2σ) = 2 * sin(σ) * cos(σ)
    term2 = a * ((x - a * np.cos(sigma)) / b) * np.sin(sigma)
    
    return term1 + term2 - y * np.cos(sigma)


def find_all_roots(x, y, a=a, b=b, num_points=1000):
    sigma_vals = np.linspace(-np.pi*1.3, np.pi*1.3, num_points)
    f_vals = sigma_func(sigma_vals, x, y, a, b)

    roots = []
    for i in range(len(f_vals) - 1):
        if np.sign(f_vals[i]) != np.sign(f_vals[i+1]):  # Sign change
            try:
                res = root_scalar(
                    sigma_func, args=(x, y, a, b),
                    bracket=[sigma_vals[i], sigma_vals[i+1]],
                    method='brentq',
                    xtol=1e-15
                )
                if res.converged:
                    root = res.root
                    # Avoid duplicates (within small tolerance)
                    if not any(np.isclose(root, r, atol=1e-5) for r in roots):
                        roots.append(root)
            except ValueError:
                continue
    return roots

def sigma(x):
    """
    Compute the parameter p that maps a given point with D on the set M.
    
    Args:
        x (array-like): The point in the domain.
        
    Returns:
        p (float array): The parameter that maps the point with D on the set M, angle in rad.
    
    """

    # Compute sigma
    sigmas = find_all_roots(x[0], x[1], a, b)

    for i in range(len(sigmas)):
        # Ensure the angle is in the range [-pi, pi]
        sigmas[i] = wrap_to_pi(sigmas[i])

    return sigmas

def D(x, sigma):
    """
    Map points in the domain to points on the manifold M using the point dependent parameter p.
    
    Args:
        x (array-like): The point in the domain.
        p (float): The parameter that maps the point with D on the set M, angle in rad.
        
    Returns:
        x_M (array-like): The point on the manifold M.
    
    """

    # 1. Compute rotation

    # Compute normal vector to ellipse through point x
    normal_vec = np.array([b*np.cos(sigma), a*np.sin(sigma), 0.0])

    m_vec = np.array([-1,0,0])

    dot_product = np.dot(m_vec[0:2], normal_vec[0:2])
    norm_m_vec = np.linalg.norm(m_vec[0:2])  
    norm_normal_vec = np.linalg.norm(normal_vec[0:2])

    cos_theta = dot_product / (norm_m_vec * norm_normal_vec)
    theta_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Clip to avoid NaN due to floating point errors

    # Adjust the angle based on the quadrant
    if normal_vec[1] > 0:
        theta_rad = -theta_rad

    # 2. Compute p(sigma)
    p = np.array([a*np.cos(sigma), b*np.sin(sigma), 0])
    
    # 3. Tranform point x to point on to the manifold M
    x_M = np.array(x, dtype=float) - p

    R_inv = np.array([[np.cos(theta_rad), np.sin(theta_rad)],
                    [-np.sin(theta_rad), np.cos(theta_rad)]])
    
    x_M[0:2] = R_inv @ x_M[0:2]

    x_M[2] = wrap_to_pi(x[2] - theta_rad)

    return x_M

print("Computing the complete CBF via equivariance properties...")

tic = time.time()
cbfModule_complete = equi_cbf.equi_cbf_synthesis(cbfModule_partially_known_cbf=cbfModule_partial,
                            D=D,
                            p=sigma,
                            domain_lower_bound=domain_lower_bound_complete,
                            domain_upper_bound=domain_upper_bound_complete,
                            discretization=discretization_complete)
toc = time.time()
print("CBF computation completed.")
print("Extra computation time for the complete CBF: ", toc-tic, " seconds.")
print("Computation of complete CBF took in total ", toc-tic+cbfModule_partial.cbf.computation_time, " seconds.")

# 3. Save the CBF module to a file
cbfModule_complete.cbf.partial_computation_time = cbfModule_partial.cbf.computation_time
cbfModule_complete.cbf.computation_time = toc-tic+cbfModule_partial.cbf.computation_time

##############################################################################################################################################
# Add correct h, cf and dynamics function to the CBF module

t0 = 0
x0 = np.array([0,0,0])
L = 1
u_min = np.array([1, -20/180*np.pi])    # [minimum speed, minimum steering angle] in [m/s, rad/s]
u_max = np.array([2, 20/180*np.pi])     # [maximum speed, maximum steering angle] in [m/s, rad/s]
myBike = Bicycle(x0=x0,
                L=1,
                u_min=u_min,
                u_max=u_max)

cbfModule_complete.dynamics = myBike
cbfModule_complete.h = h
cbfModule_complete.terminal_condition = None
##############################################################################################################################################

# 5. Save the CBF module of the complete CBF to a file
cbfModule_complete.save(cbf_complete_file_name, folder_name="Data")

print("CBF computation finished.")



