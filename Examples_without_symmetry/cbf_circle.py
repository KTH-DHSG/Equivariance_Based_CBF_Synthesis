"""

    Equivariance based CBF synthesis for the bicycle model using the knowledge over a partially known CBF. 
    The constraint under consideration is a 

    >> circular obstacle

    Adrian Wiltz, 2025

"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
sys.path.append(str('Predictive_CBF_synthesis_Toolbox'))

import numpy as np
from Dynamics.Bicycle import Bicycle
from CBF.CBFmodule import CBFmodule
import CBF.CBFcomputation as CBFcomputation
from Equivariance_Module import multi_equi_cbf
from math_aux.math_aux import wrap_to_pi
import casadi as ca
import time

import warnings

########################################################################################

cbf_module_filename = "2025-06-17_16-41-39_cbf_straight_line_partial.json"

cbf_module_folder_path = os.path.abspath(r'Examples_without_symmetry/Data_cbf_partial')

cbf_complete_file_name = 'cbf_circle.json'

########################################################################################
# load partially known cbf

cbfModule_partial = CBFmodule()
cbfModule_partial.load(cbf_module_filename, cbf_module_folder_path)

########################################################################################
# Constraint specifications

radius = 4

def h(x):
    """
    Constraint function h for the circle with radius 4.
    
    Args:
        x (array-like): The point in the domain.
        
    Returns:
        h_value (float): The value of the constraint function.
    
    """

    radius = 4

    return np.linalg.norm(x[0:2]) - radius

########################################################################################
# Some CBF settings

domain_lower_bound_complete = np.array([-10,-10,-np.pi])
domain_upper_bound_complete = np.array([10,10,np.pi])
discretization_complete = np.array([21,21,41])

########################################################################################
# Define transformation and parameter function

def eval_condition1(x):
    """
    Evaluate the condition for the first transformation.
    
    Args:
        x (array-like): The point in the domain.
        
    Returns:
        bool: True if the condition is satisfied, False otherwise.
    
    """
    
    return True

def p1(x):
    """
    Compute the parameter p that maps a given point with D on the set M.
    
    Args:
        x (array-like): The point in the domain.
        
    Returns:
        p (float): The parameter that maps the point with D on the set M, angle in rad.
    
    """
    
    m_vec = np.array([-1,0,0])

    dot_product = np.dot(m_vec[0:2], x[0:2])
    norm_m_vec = np.linalg.norm(m_vec[0:2])  
    norm_x = np.linalg.norm(x[0:2])
    
    if norm_x == 0: 
        theta_rad = 0.0
    else:
        cos_theta = dot_product / (norm_m_vec * norm_x)
        theta_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Clip to avoid NaN due to floating point errors

    # Adjust the angle based on the quadrant
    if x[1] > 0:
        theta_rad = -theta_rad

    return theta_rad

def D1(x, p):
    """
    Map points in the domain to points on the manifold M using the point dependent parameter p.
    
    Args:
        x (array-like): The point in the domain.
        p (float): The parameter that maps the point with D on the set M, angle in rad.
        
    Returns:
        x_M (array-like): The point on the manifold M.
    
    """
    
    x_M = np.array(x)

    R_inv = np.array([[np.cos(p), np.sin(p)],
                    [-np.sin(p), np.cos(p)]])
    
    x_M[0:2] = R_inv @ x[0:2]

    x_M[0:2] = x_M[0:2] + np.array([radius,0])

    x_M[2] = wrap_to_pi(x[2] - p)

    return x_M

def eval_condition2(x):
    """
    Evaluate the condition for the first transformation.
    
    Args:
        x (array-like): The point in the domain.
        
    Returns:
        bool: True if the condition is satisfied, False otherwise.
    
    """
    
    return True

def p2(x):
    """
    Compute the parameter p that maps a given point with D on the set M.

    Rotation by additional pi.
    
    Args:
        x (array-like): The point in the domain.
        
    Returns:
        p (float): The parameter that maps the point with D on the set M, angle in rad.
    
    """
    
    m_vec = np.array([-1,0,0])

    dot_product = np.dot(m_vec[0:2], x[0:2])
    norm_m_vec = np.linalg.norm(m_vec[0:2])  
    norm_x = np.linalg.norm(x[0:2])
    
    if norm_x == 0: 
        theta_rad = np.pi
    else:
        cos_theta = dot_product / (norm_m_vec * norm_x)
        theta_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0)) + np.pi  # Clip to avoid NaN due to floating point errors

    # Adjust the angle based on the quadrant
    if x[1] > 0:
        theta_rad = -theta_rad

    return theta_rad

def D2(x, p):
    """
    Map points in the domain to points on the manifold M using the point dependent parameter p.
    
    Args:
        x (array-like): The point in the domain.
        p (float): The parameter that maps the point with D on the set M, angle in rad.
        
    Returns:
        x_M (array-like): The point on the manifold M.
    
    """
    
    x_M = np.array(x)

    R_inv = np.array([[np.cos(p), np.sin(p)],
                    [-np.sin(p), np.cos(p)]])
    
    x_M[0:2] = R_inv @ x[0:2]

    x_M[0:2] = x_M[0:2] + np.array([radius,0])

    x_M[2] = wrap_to_pi(x[2] - p)

    return x_M

eval_conditions = [eval_condition1, eval_condition2]
p_list = [p1, p2]
D_list = [D1, D2]

########################################################################################
# Compute the complete CBF based on equivariances without exploiting symmetry

tic = time.time()

cbfModule_complete = multi_equi_cbf.multi_equi_cbf_synthesis(cbfModule_partially_known_cbf=cbfModule_partial,
                            D_list=D_list,
                            p_list=p_list,
                            eval_condtions=eval_conditions,
                            domain_lower_bound=domain_lower_bound_complete,
                            domain_upper_bound=domain_upper_bound_complete,
                            discretization=discretization_complete)
toc = time.time()
print("Extra computation time for the complete CBF: ", toc-tic, " seconds.")
print("Computation of complete CBF took in total ", toc-tic+cbfModule_partial.cbf.computation_time, " seconds.")

# Save the CBF module to a file
cbfModule_complete.cbf.partial_computation_time = cbfModule_partial.cbf.computation_time
cbfModule_complete.cbf.computation_time = toc-tic+cbfModule_partial.cbf.computation_time

##############################################################################################################################################
# Add correct h, cf and dynamics function to the CBF module

def cf(x):
    """Terminal constraint function for casadi type arguments"""

    h_grad = np.array([-1,0])
    turning_radius = 2.8
    orientation = ca.vertcat(ca.cos(x[2]), ca.sin(x[2]))

    return ca.vertcat(ca.dot(h_grad, orientation), -x[0] - 2*turning_radius)

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
cbfModule_complete.terminal_condition = cf
##############################################################################################################################################

# Save the CBF module of the complete CBF to a file
cbfModule_complete.save(cbf_complete_file_name, folder_name="Data")

print("CBF computation finished.")



