"""

    Equivariance based CBF synthesis for the bicycle model using the knowledge over a partially known CBF. 
    The constraint under consideration is a 

    >> square as obstacle

    Adrian Wiltz, 2025

"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import numpy as np
from Dynamics.Bicycle import Bicycle
from CBF.CBFmodule import CBFmodule
import CBF.CBFcomputation as CBFcomputation
from Equivariance_Module import multi_equi_cbf
from math_aux.math_aux import wrap_to_pi
import casadi as ca
import time

########################################################################################

cbf_module_filename = "2025-06-17_16-41-39_cbf_straight_line_partial.json"

cbf_module_folder_path = r'Examples_without_symmetry/Data_cbf_partial'

cbf_complete_file_name = 'cbf_square.json'

########################################################################################
# load partially known cbf

cbfModule_partial = CBFmodule()
cbfModule_partial.load(cbf_module_filename, cbf_module_folder_path)

########################################################################################
# Constraint specifications

length = 4

def h(x):
    """
    Constraint function h for the square with side length 4.
    
    Args:
        x (array-like): The point in the domain.
        
    Returns:
        h_value (float): The value of the constraint function.
    
    """

    length = 4
    
    return np.maximum(np.abs(x[0]) - length/2, np.abs(x[1]) - length/2)

########################################################################################
# Some CBF settings

domain_lower_bound_complete = np.array([-10,-10,-np.pi])
domain_upper_bound_complete = np.array([10,10,np.pi])
discretization_complete = np.array([21,21,41])

########################################################################################
# 1. Define transformation and parameter function and evaluation conditions

def eval_condition1(x):
    """
    Evaluate the condition for the first transformation.
    
    Args:
        x (array-like): The point in the domain.
        
    Returns:
        bool: True if the condition is satisfied, False otherwise.
    
    """
    
    return x[0] <= 0 

def p1(x):
    """
    Compute the parameter p that maps a given point with D on the set M.
    
    Args:
        x (array-like): The point in the domain.
        
    Returns:
        p (float): The parameter that maps the point with D on the set M, angle in rad.
    
    """
    
    return x[1]

def D1(x, p):
    """
    Map points in the domain to points on the manifold M using the point dependent parameter p.
    
    Args:
        x (array-like): The point in the domain.
        p (float): The parameter that maps the point with D on the set M, angle in rad.
        
    Returns:
        x_M (array-like): The point on the manifold M.
    
    """
    
    x_M = np.array(x, dtype=float)

    x_M[0:2] = x_M[0:2] - np.array([-length/2,p])

    return x_M

def eval_condition2(x):
    """
    Evaluate the condition for the first transformation.
    
    Args:
        x (array-like): The point in the domain.
        
    Returns:
        bool: True if the condition is satisfied, False otherwise.
    
    """
    
    return x[1] >= 0

def p2(x):
    """
    Compute the parameter p that maps a given point with D on the set M.
    
    Args:
        x (array-like): The point in the domain.
        
    Returns:
        p (float): The parameter that maps the point with D on the set M, angle in rad.
    
    """
    
    return x[0]

def D2(x, p):
    """
    Map points in the domain to points on the manifold M using the point dependent parameter p.
    
    Args:
        x (array-like): The point in the domain.
        p (float): The parameter that maps the point with D on the set M, angle in rad.
        
    Returns:
        x_M (array-like): The point on the manifold M.
    
    """
    
    x_M = np.array(x, dtype=float)  

    x_M[0] = x[0] - p

    rot_angle = -np.pi/2
    R_inv = np.array([[np.cos(rot_angle), np.sin(rot_angle)],
                    [-np.sin(rot_angle), np.cos(rot_angle)]])

    x_M[0:2] = R_inv @ x_M[0:2]

    x_M[0:2] = x_M[0:2] - np.array([-length/2,0])

    x_M[2] = wrap_to_pi(x[2] - rot_angle)

    return x_M

def eval_condition3(x):
    """
    Evaluate the condition for the first transformation.
    
    Args:
        x (array-like): The point in the domain.
        
    Returns:
        bool: True if the condition is satisfied, False otherwise.
    
    """
    
    return x[0] >= 0

def p3(x):
    """
    Compute the parameter p that maps a given point with D on the set M.
    
    Args:
        x (array-like): The point in the domain.
        
    Returns:
        p (float): The parameter that maps the point with D on the set M, angle in rad.
    
    """
    
    return x[1]

def D3(x, p):
    """
    Map points in the domain to points on the manifold M using the point dependent parameter p.
    
    Args:
        x (array-like): The point in the domain.
        p (float): The parameter that maps the point with D on the set M, angle in rad.
        
    Returns:
        x_M (array-like): The point on the manifold M.
    
    """
    
    x_M = np.array(x, dtype=float)

    x_M[1] = x_M[1] - p

    rot_angle = -np.pi
    R_inv = np.array([[np.cos(rot_angle), np.sin(rot_angle)],
                    [-np.sin(rot_angle), np.cos(rot_angle)]])

    x_M[0:2] = R_inv @ x[0:2]

    x_M[0:2] = x_M[0:2] - np.array([-length/2,0])

    x_M[2] = wrap_to_pi(x[2] - rot_angle)

    return x_M

def eval_condition4(x):
    """
    Evaluate the condition for the first transformation.
    
    Args:
        x (array-like): The point in the domain.
        
    Returns:
        bool: True if the condition is satisfied, False otherwise.
    
    """
    
    return x[1] <= 0

def p4(x):
    """
    Compute the parameter p that maps a given point with D on the set M.
    
    Args:
        x (array-like): The point in the domain.
        
    Returns:
        p (float): The parameter that maps the point with D on the set M, angle in rad.
    
    """
    
    return x[0]

def D4(x, p):
    """
    Map points in the domain to points on the manifold M using the point dependent parameter p.
    
    Args:
        x (array-like): The point in the domain.
        p (float): The parameter that maps the point with D on the set M, angle in rad.
        
    Returns:
        x_M (array-like): The point on the manifold M.
    
    """

    x_M = np.array(x, dtype=float)
    
    x_M[0] = x[0] - p

    rot_angle = np.pi/2
    R_inv = np.array([[np.cos(rot_angle), np.sin(rot_angle)],
                    [-np.sin(rot_angle), np.cos(rot_angle)]])

    x_M[0:2] = R_inv @ x_M[0:2]

    x_M[0:2] = x_M[0:2] - np.array([-length/2,0])

    x_M[2] = wrap_to_pi(x[2] - rot_angle)

    return x_M

def eval_condition5(x):
    """
    Evaluate the condition for the first transformation.
    
    Args:
        x (array-like): The point in the domain.
        
    Returns:
        bool: True if the condition is satisfied, False otherwise.
    
    """
    
    # true if point in the upper left corner of the square or
    # if the point is located within the square
    return (x[0] < -length/2 and x[1] > length/2) or (x[0] > -length/2 and x[0] < length/2 and x[1] < length/2 and x[1] > -length/2)

def p5(x):
    """
    Compute the parameter p that maps a given point with D on the set M.
    
    Args:
        x (array-like): The point in the domain.
        
    Returns:
        p (float): The parameter that maps the point with D on the set M, angle in rad.
    
    """
    
    x_tmp = np.array(x) + np.array([length/2,-length/2,0])

    m_vec = np.array([-1,0,0])

    dot_product = np.dot(m_vec[0:2], x_tmp[0:2])
    norm_m_vec = np.linalg.norm(m_vec[0:2])  
    norm_x = np.linalg.norm(x_tmp[0:2])

    cos_theta = dot_product / (norm_m_vec * norm_x)
    theta_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Clip to avoid NaN due to floating point errors

    # Check if the angle is in the correct quadrant, otherwise ignore it
    if theta_rad < 0 or theta_rad > np.pi/2:
        theta_rad = np.nan

    # Adjust the angle based on the quadrant
    theta_rad = -theta_rad

    return theta_rad

def D5(x, p):
    """
    Map points in the domain to points on the manifold M using the point dependent parameter p.
    
    Args:
        x (array-like): The point in the domain.
        p (float): The parameter that maps the point with D on the set M, angle in rad.
        
    Returns:
        x_M (array-like): The point on the manifold M.
    
    """
    
    x_M = np.array(x, dtype=float) + np.array([length/2,-length/2,0])

    R_inv = np.array([[np.cos(p), np.sin(p)],
                    [-np.sin(p), np.cos(p)]])

    x_M[0:2] = R_inv @ x_M[0:2]

    x_M[2] = wrap_to_pi(x_M[2] - p)

    return x_M

def eval_condition6(x):
    """
    Evaluate the condition for the first transformation.
    
    Args:
        x (array-like): The point in the domain.
        
    Returns:
        bool: True if the condition is satisfied, False otherwise.
    
    """
    
    # true if point in the upper right corner of the square or
    # if the point is located within the square
    return (x[0] > length/2 and x[1] > length/2) or (x[0] > -length/2 and x[0] < length/2 and x[1] < length/2 and x[1] > -length/2)

def p6(x):
    """
    Compute the parameter p that maps a given point with D on the set M.
    
    Args:
        x (array-like): The point in the domain.
        
    Returns:
        p (float): The parameter that maps the point with D on the set M, angle in rad.
    
    """
    
    x_tmp = np.array(x, dtype=float) + np.array([-length/2,-length/2,0])

    m_vec = np.array([-1,0,0])

    dot_product = np.dot(m_vec[0:2], x_tmp[0:2])
    norm_m_vec = np.linalg.norm(m_vec[0:2])  
    norm_x = np.linalg.norm(x_tmp[0:2])

    cos_theta = dot_product / (norm_m_vec * norm_x)
    theta_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Clip to avoid NaN due to floating point errors

    # Check if the angle is in the correct quadrant, otherwise ignore it
    if theta_rad < np.pi/2 or theta_rad > np.pi:
        theta_rad = np.nan

    # Adjust the angle based on the quadrant
    theta_rad = -theta_rad

    return theta_rad

def D6(x, p):
    """
    Map points in the domain to points on the manifold M using the point dependent parameter p.
    
    Args:
        x (array-like): The point in the domain.
        p (float): The parameter that maps the point with D on the set M, angle in rad.
        
    Returns:
        x_M (array-like): The point on the manifold M.
    
    """
    
    x_M = np.array(x, dtype=float) + np.array([-length/2,-length/2,0])

    R_inv = np.array([[np.cos(p), np.sin(p)],
                    [-np.sin(p), np.cos(p)]])

    x_M[0:2] = R_inv @ x_M[0:2]

    x_M[2] = wrap_to_pi(x_M[2] - p)

    return x_M

def eval_condition7(x):
    """
    Evaluate the condition for the first transformation.
    
    Args:
        x (array-like): The point in the domain.
        
    Returns:
        bool: True if the condition is satisfied, False otherwise.
    
    """
    
    # true if point in the lower right corner of the square or
    # if the point is located within the square
    return (x[0] > length/2 and x[1] < -length/2) or (x[0] > -length/2 and x[0] < length/2 and x[1] < length/2 and x[1] > -length/2)

def p7(x):
    """
    Compute the parameter p that maps a given point with D on the set M.
    
    Args:
        x (array-like): The point in the domain.
        
    Returns:
        p (float): The parameter that maps the point with D on the set M, angle in rad.
    
    """
    
    x_tmp = np.array(x, dtype = float) + np.array([-length/2,length/2,0])

    m_vec = np.array([-1,0,0])

    dot_product = np.dot(m_vec[0:2], x_tmp[0:2])
    norm_m_vec = np.linalg.norm(m_vec[0:2])  
    norm_x = np.linalg.norm(x_tmp[0:2])

    cos_theta = dot_product / (norm_m_vec * norm_x)
    theta_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Clip to avoid NaN due to floating point errors

    # Check if the angle is in the correct quadrant, otherwise ignore it
    if theta_rad < np.pi/2 or theta_rad > np.pi:
        theta_rad = np.nan

    return theta_rad

def D7(x, p):
    """
    Map points in the domain to points on the manifold M using the point dependent parameter p.
    
    Args:
        x (array-like): The point in the domain.
        p (float): The parameter that maps the point with D on the set M, angle in rad.
        
    Returns:
        x_M (array-like): The point on the manifold M.
    
    """
    
    x_M = np.array(x, dtype=float) + np.array([-length/2,length/2,0])

    R_inv = np.array([[np.cos(p), np.sin(p)],
                    [-np.sin(p), np.cos(p)]])

    x_M[0:2] = R_inv @ x_M[0:2]

    x_M[2] = wrap_to_pi(x_M[2] - p)

    return x_M

def eval_condition8(x):
    """
    Evaluate the condition for the first transformation.
    
    Args:
        x (array-like): The point in the domain.
        
    Returns:
        bool: True if the condition is satisfied, False otherwise.
    
    """
    
    # true if point in the lower left corner of the square or
    # if the point is located within the square
    return (x[0] < -length/2 and x[1] < -length/2) or (x[0] > -length/2 and x[0] < length/2 and x[1] < length/2 and x[1] > -length/2)

def p8(x):
    """
    Compute the parameter p that maps a given point with D on the set M.
    
    Args:
        x (array-like): The point in the domain.
        
    Returns:
        p (float): The parameter that maps the point with D on the set M, angle in rad.
    
    """
    
    x_tmp = np.array(x, dtype=float) + np.array([length/2,length/2,0])

    m_vec = np.array([-1,0,0])

    dot_product = np.dot(m_vec[0:2], x_tmp[0:2])
    norm_m_vec = np.linalg.norm(m_vec[0:2])  
    norm_x = np.linalg.norm(x_tmp[0:2])

    cos_theta = dot_product / (norm_m_vec * norm_x)
    theta_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Clip to avoid NaN due to floating point errors

    # Check if the angle is in the correct quadrant, otherwise ignore it
    if theta_rad < 0 or theta_rad > np.pi/2:
        theta_rad = np.nan

    return theta_rad

def D8(x, p):
    """
    Map points in the domain to points on the manifold M using the point dependent parameter p.
    
    Args:
        x (array-like): The point in the domain.
        p (float): The parameter that maps the point with D on the set M, angle in rad.
        
    Returns:
        x_M (array-like): The point on the manifold M.
    
    """
    
    x_M = np.array(x, dtype=float) + np.array([length/2,length/2,0])

    R_inv = np.array([[np.cos(p), np.sin(p)],
                    [-np.sin(p), np.cos(p)]])

    x_M[0:2] = R_inv @ x_M[0:2]

    x_M[2] = wrap_to_pi(x_M[2] - p)

    return x_M

eval_conditions = [eval_condition1, eval_condition2, eval_condition3, eval_condition4,
                   eval_condition5, eval_condition6, eval_condition7, eval_condition8]
p_list = [p1, p2, p3, p4, p5, p6, p7, p8]
D_list = [D1, D2, D3, D4, D5, D6, D7, D8]

########################################################################################
# 2. Synthesize the complete CBF module using the partially known CBF module and the transformations

print("CBF computation based on equivariances started...")

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

########################################################################################
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

cbfModule_complete.save(cbf_complete_file_name, folder_name="Data")

print("CBF computation finished.")



