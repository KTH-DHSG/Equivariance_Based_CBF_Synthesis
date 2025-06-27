"""

    Equivariance based CBF synthesis for the bicycle model using the knowledge over a partially known CBF. 
    The constraint under consideration is a 

    >> non-convex advanced obstacle

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

cbf_module_inner_circle_filename = "2025-06-17_16-42-22_cbf_inner_circle_partial.json"
cbf_module_inner_circle_folder_path = r'Examples_without_symmetry/Data_cbf_partial'

cbf_complete_file_name = 'cbf_nonconvex_obstacle.json'

########################################################################################
# load partially known cbf

cbfModule_inner_circle_partial = CBFmodule()
cbfModule_inner_circle_partial.load(cbf_module_inner_circle_filename, cbf_module_inner_circle_folder_path)

########################################################################################
# Constraint specifications

cbf_radius = cbfModule_inner_circle_partial.radius
turning_radius = cbfModule_inner_circle_partial.turning_radius

radius_1 = 7.5
# l_corner = 30.0
l_corner = 18
l1 = 1/np.sqrt(2) * radius_1
l3 = radius_1 - l1

l2 = l_corner - l1
radius_2 = np.sqrt(2)*l2
l4 = radius_2 - l2
l_center = l_corner + l1 + l2

radius_0 = l_corner + l1 - l4 - 2*radius_1

point_0 = np.array([0.0,0.0])
point_1 = np.array([-l_corner,l_corner])
point_2 = np.array([l_corner,l_corner])
point_3 = np.array([l_corner,-l_corner])
point_4 = np.array([-l_corner,-l_corner])
point_5 = np.array([-l_center,0.0])
point_6 = np.array([0.0,l_center])
point_7 = np.array([l_center,0.0])
point_8 = np.array([0.0,-l_center])

def h(x):
    """
    Constraint function h for the square with side length 4.
    
    Args:
        x (array-like): The point in the domain.
        
    Returns:
        h_value (float): The value of the constraint function.
    
    """

    radius_1 = 7.5
    # l_corner = 30.0
    l_corner = 18
    l1 = 1/np.sqrt(2) * radius_1
    l3 = radius_1 - l1

    l2 = l_corner - l1
    radius_2 = np.sqrt(2)*l2
    l4 = radius_2 - l2
    l_center = l_corner + l1 + l2

    radius_0 = l_corner + l1 - l4 - 2*radius_1

    point_0 = np.array([0.0,0.0])
    point_1 = np.array([-l_corner,l_corner])
    point_2 = np.array([l_corner,l_corner])
    point_3 = np.array([l_corner,-l_corner])
    point_4 = np.array([-l_corner,-l_corner])
    point_5 = np.array([-l_center,0.0])
    point_6 = np.array([0.0,l_center])
    point_7 = np.array([l_center,0.0])
    point_8 = np.array([0.0,-l_center])

    h_0 = np.sqrt((x[0] - point_0[0])**2 + (x[1]-point_0[1])**2) - radius_0
    h_1 = -np.sqrt((x[0] - point_1[0])**2 + (x[1]-point_1[1])**2) + radius_1
    h_2 = -np.sqrt((x[0] - point_2[0])**2 + (x[1]-point_2[1])**2) + radius_1
    h_3 = -np.sqrt((x[0] - point_3[0])**2 + (x[1]-point_3[1])**2) + radius_1
    h_4 = -np.sqrt((x[0] - point_4[0])**2 + (x[1]-point_4[1])**2) + radius_1
    h_5 = np.sqrt((x[0] - point_5[0])**2 + (x[1]-point_5[1])**2) - radius_2
    h_6 = np.sqrt((x[0] - point_6[0])**2 + (x[1]-point_6[1])**2) - radius_2
    h_7 = np.sqrt((x[0] - point_7[0])**2 + (x[1]-point_7[1])**2) - radius_2
    h_8 = np.sqrt((x[0] - point_8[0])**2 + (x[1]-point_8[1])**2) - radius_2

    h_tmp = np.min([h_0, h_5, h_6, h_7, h_8])

    if np.dot(np.array([-1.0,1.0]),x[0:2] - point_1) >= 0:
        h_tmp = np.min([h_tmp, h_1])

    if np.dot(np.array([1.0,1.0]),x[0:2] - point_2) >= 0:
        h_tmp = np.min([h_tmp, h_2])

    if np.dot(np.array([1.0,-1.0]),x[0:2] - point_3) >= 0:
        h_tmp = np.min([h_tmp, h_3])

    if np.dot(np.array([-1.0,-1.0]),x[0:2] - point_4) >= 0:
        h_tmp = np.min([h_tmp, h_4])

    return h_tmp


########################################################################################
# Some CBF settings

domain_lower_bound_complete = np.array([-l_corner-17,-l_corner-17,-np.pi])
domain_upper_bound_complete = np.array([l_corner+17,l_corner+17,np.pi])
discretization_complete = np.array([250,250,41])

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
    
    return np.dot(np.array([-1.0,1.0]),x[0:2] - point_1) >= 0

def p1(x):
    """
    Compute the parameter p that maps a given point with D on the set M.
    
    Args:
        x (array-like): The point in the domain.
        
    Returns:
        p (float): The parameter that maps the point with D on the set M, angle in rad.
    
    """
    
    x_tmp = np.array(x, dtype=float) - np.array([point_1[0],point_1[1],0])

    m_vec = np.array([-1,0,0])

    dot_product = np.dot(m_vec[0:2], x_tmp[0:2])
    norm_m_vec = np.linalg.norm(m_vec[0:2])  
    norm_x = np.linalg.norm(x_tmp[0:2])

    cos_theta = dot_product / (norm_m_vec * norm_x)
    theta_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Clip to avoid NaN due to floating point errors

    # Adjust the angle based on the quadrant
    if x_tmp[1] > 0:
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
    
    x_M = np.array(x, dtype=float) - np.array([point_1[0],point_1[1],0])

    R_inv = np.array([[np.cos(p), np.sin(p)],
                    [-np.sin(p), np.cos(p)]])
    
    x_M[0:2] = R_inv @ x_M[0:2]

    x_M[0:2] = x_M[0:2] + np.array([radius_1-cbf_radius,0])

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
    
    return np.dot(np.array([1.0,1.0]),x[0:2] - point_2) >= 0

def p2(x):
    """
    Compute the parameter p that maps a given point with D on the set M.
    
    Args:
        x (array-like): The point in the domain.
        
    Returns:
        p (float): The parameter that maps the point with D on the set M, angle in rad.
    
    """
    
    x_tmp = np.array(x, dtype=float) - np.array([point_2[0],point_2[1],0])

    m_vec = np.array([-1,0,0])

    dot_product = np.dot(m_vec[0:2], x_tmp[0:2])
    norm_m_vec = np.linalg.norm(m_vec[0:2])  
    norm_x = np.linalg.norm(x_tmp[0:2])

    cos_theta = dot_product / (norm_m_vec * norm_x)
    theta_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Clip to avoid NaN due to floating point errors

    # Adjust the angle based on the quadrant
    if x_tmp[1] > 0:
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
    
    x_M = np.array(x, dtype=float) - np.array([point_2[0],point_2[1],0])

    R_inv = np.array([[np.cos(p), np.sin(p)],
                    [-np.sin(p), np.cos(p)]])
    
    x_M[0:2] = R_inv @ x_M[0:2]

    x_M[0:2] = x_M[0:2] + np.array([radius_1-cbf_radius,0])

    x_M[2] = wrap_to_pi(x[2] - p)

    return x_M

def eval_condition3(x):
    """
    Evaluate the condition for the first transformation.
    
    Args:
        x (array-like): The point in the domain.
        
    Returns:
        bool: True if the condition is satisfied, False otherwise.
    
    """
    
    return np.dot(np.array([1.0,-1.0]),x[0:2] - point_3) >= 0

def p3(x):
    """
    Compute the parameter p that maps a given point with D on the set M.
    
    Args:
        x (array-like): The point in the domain.
        
    Returns:
        p (float): The parameter that maps the point with D on the set M, angle in rad.
    
    """
    
    x_tmp = np.array(x, dtype=float) - np.array([point_3[0],point_3[1],0])

    m_vec = np.array([-1,0,0])

    dot_product = np.dot(m_vec[0:2], x_tmp[0:2])
    norm_m_vec = np.linalg.norm(m_vec[0:2])  
    norm_x = np.linalg.norm(x_tmp[0:2])

    cos_theta = dot_product / (norm_m_vec * norm_x)
    theta_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Clip to avoid NaN due to floating point errors

    # Adjust the angle based on the quadrant
    if x_tmp[1] > 0:
        theta_rad = -theta_rad

    return theta_rad

def D3(x, p):
    """
    Map points in the domain to points on the manifold M using the point dependent parameter p.
    
    Args:
        x (array-like): The point in the domain.
        p (float): The parameter that maps the point with D on the set M, angle in rad.
        
    Returns:
        x_M (array-like): The point on the manifold M.
    
    """
    
    x_M = np.array(x, dtype=float) - np.array([point_3[0],point_3[1],0])

    R_inv = np.array([[np.cos(p), np.sin(p)],
                    [-np.sin(p), np.cos(p)]])
    
    x_M[0:2] = R_inv @ x_M[0:2]

    x_M[0:2] = x_M[0:2] + np.array([radius_1-cbf_radius,0])

    x_M[2] = wrap_to_pi(x[2] - p)

    return x_M

def eval_condition4(x):
    """
    Evaluate the condition for the first transformation.
    
    Args:
        x (array-like): The point in the domain.
        
    Returns:
        bool: True if the condition is satisfied, False otherwise.
    
    """
    
    return np.dot(np.array([-1.0,-1.0]),x[0:2] - point_4) >= 0

def p4(x):
    """
    Compute the parameter p that maps a given point with D on the set M.
    
    Args:
        x (array-like): The point in the domain.
        
    Returns:
        p (float): The parameter that maps the point with D on the set M, angle in rad.
    
    """
    
    x_tmp = np.array(x, dtype=float) - np.array([point_4[0],point_4[1],0])

    m_vec = np.array([-1,0,0])

    dot_product = np.dot(m_vec[0:2], x_tmp[0:2])
    norm_m_vec = np.linalg.norm(m_vec[0:2])  
    norm_x = np.linalg.norm(x_tmp[0:2])

    cos_theta = dot_product / (norm_m_vec * norm_x)
    theta_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Clip to avoid NaN due to floating point errors

    # Adjust the angle based on the quadrant
    if x_tmp[1] > 0:
        theta_rad = -theta_rad

    return theta_rad

def D4(x, p):
    """
    Map points in the domain to points on the manifold M using the point dependent parameter p.
    
    Args:
        x (array-like): The point in the domain.
        p (float): The parameter that maps the point with D on the set M, angle in rad.
        
    Returns:
        x_M (array-like): The point on the manifold M.
    
    """
    
    x_M = np.array(x, dtype=float) - np.array([point_4[0],point_4[1],0])

    R_inv = np.array([[np.cos(p), np.sin(p)],
                    [-np.sin(p), np.cos(p)]])
    
    x_M[0:2] = R_inv @ x_M[0:2]

    x_M[0:2] = x_M[0:2] + np.array([radius_1-cbf_radius,0])

    x_M[2] = wrap_to_pi(x[2] - p)

    return x_M

def eval_condition5(x):
    """
    Evaluate the condition for the first transformation.
    
    Args:
        x (array-like): The point in the domain.
        
    Returns:
        bool: True if the condition is satisfied, False otherwise.
    
    """

    condition_1 = np.dot(np.array([1.0,-1.0]),x[0:2] - point_1) >= 0 and np.dot(np.array([1.0,1.0]),x[0:2] - point_4) >= 0 and np.sqrt((x[0] - point_5[0])**2 + (x[1]-point_5[1])**2) < radius_2 + cbf_radius

    condition_2 = x[0] < point_5[0] + l2 and x[1] > point_5[1] - l2 and x[1] < point_5[1] + l2
    
    return condition_1 or condition_2

def p5(x):
    """
    Compute the parameter p that maps a given point with D on the set M.
    
    Args:
        x (array-like): The point in the domain.
        
    Returns:
        p (float): The parameter that maps the point with D on the set M, angle in rad.
    
    """
    
    x_tmp = np.array(x, dtype=float) - np.array([point_5[0],point_5[1],0])

    m_vec = np.array([1,0,0])

    dot_product = np.dot(m_vec[0:2], x_tmp[0:2])
    norm_m_vec = np.linalg.norm(m_vec[0:2])  
    norm_x = np.linalg.norm(x_tmp[0:2])

    cos_theta = dot_product / (norm_m_vec * norm_x)
    theta_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Clip to avoid NaN due to floating point errors

    # Adjust the angle based on the quadrant
    if x_tmp[1] < 0:
        theta_rad = -theta_rad

    # Check if the angle is in the correct range, otherwise ignore it
    if not (theta_rad >= -1/4*np.pi and theta_rad <= 1/4*np.pi):
        theta_rad = np.nan 

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
    
    x_M = np.array(x, dtype=float) - np.array([point_5[0],point_5[1],0])

    R_inv = np.array([[np.cos(p), np.sin(p)],
                    [-np.sin(p), np.cos(p)]])
    
    x_M[0:2] = R_inv @ x_M[0:2]

    x_M[0:2] = x_M[0:2] + np.array([-radius_2-cbf_radius,0])

    x_M[2] = wrap_to_pi(x[2] - p)

    return x_M

def eval_condition6(x):
    """
    Evaluate the condition for the first transformation.
    
    Args:
        x (array-like): The point in the domain.
        
    Returns:
        bool: True if the condition is satisfied, False otherwise.
    
    """

    condition_1 = np.dot(np.array([1.0,-1.0]),x[0:2] - point_1) >= 0 and np.dot(np.array([-1.0,-1.0]),x[0:2] - point_2) >= 0 and np.sqrt((x[0] - point_6[0])**2 + (x[1]-point_6[1])**2) < radius_2 + cbf_radius

    condition_2 = x[1] < point_6[1] - l2 and x[0] > point_6[0] - l2 and x[0] < point_6[0] + l2
    
    return condition_1 or condition_2

def p6(x):
    """
    Compute the parameter p that maps a given point with D on the set M.
    
    Args:
        x (array-like): The point in the domain.
        
    Returns:
        p (float): The parameter that maps the point with D on the set M, angle in rad.
    
    """
    
    x_tmp = np.array(x, dtype=float) - np.array([point_6[0],point_6[1],0])

    m_vec = np.array([1,0,0])

    dot_product = np.dot(m_vec[0:2], x_tmp[0:2])
    norm_m_vec = np.linalg.norm(m_vec[0:2])  
    norm_x = np.linalg.norm(x_tmp[0:2])

    cos_theta = dot_product / (norm_m_vec * norm_x)
    theta_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Clip to avoid NaN due to floating point errors

    # Adjust the angle based on the quadrant
    theta_rad = -theta_rad

    # Check if the angle is in the correct range, otherwise ignore it
    if not (theta_rad >= -3/4*np.pi and theta_rad <= -1/4*np.pi):
        theta_rad = np.nan 

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
    
    x_M = np.array(x, dtype=float) - np.array([point_6[0],point_6[1],0])

    R_inv = np.array([[np.cos(p), np.sin(p)],
                    [-np.sin(p), np.cos(p)]])
    
    x_M[0:2] = R_inv @ x_M[0:2]

    x_M[0:2] = x_M[0:2] + np.array([-radius_2-cbf_radius,0])

    x_M[2] = wrap_to_pi(x[2] - p)

    return x_M

def eval_condition7(x):
    """
    Evaluate the condition for the first transformation.
    
    Args:
        x (array-like): The point in the domain.
        
    Returns:
        bool: True if the condition is satisfied, False otherwise.
    
    """

    condition_1 = np.dot(np.array([-1.0,1.0]),x[0:2] - point_3) >= 0 and np.dot(np.array([-1.0,-1.0]),x[0:2] - point_2) >= 0 and np.sqrt((x[0] - point_7[0])**2 + (x[1]-point_7[1])**2) < radius_2 + cbf_radius

    condition_2 = x[0] > point_7[0] - l2 and x[1] > point_7[0] - l2 and x[1] < point_7[1] + l2
    
    return condition_1 or condition_2

def p7(x):
    """
    Compute the parameter p that maps a given point with D on the set M.
    
    Args:
        x (array-like): The point in the domain.
        
    Returns:
        p (float): The parameter that maps the point with D on the set M, angle in rad.
    
    """
    
    x_tmp = np.array(x, dtype=float) - np.array([point_7[0],point_7[1],0])

    m_vec = np.array([1,0,0])

    dot_product = np.dot(m_vec[0:2], x_tmp[0:2])
    norm_m_vec = np.linalg.norm(m_vec[0:2])  
    norm_x = np.linalg.norm(x_tmp[0:2])

    cos_theta = dot_product / (norm_m_vec * norm_x)
    theta_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Clip to avoid NaN due to floating point errors

    # Adjust the angle based on the quadrant
    if x_tmp[1] < 0:
        theta_rad = -theta_rad

    # Check if the angle is in the correct range, otherwise ignore it
    if not (theta_rad >= 3/4*np.pi or theta_rad <= -3/4*np.pi):
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
    
    x_M = np.array(x, dtype=float) - np.array([point_7[0],point_7[1],0])

    R_inv = np.array([[np.cos(p), np.sin(p)],
                    [-np.sin(p), np.cos(p)]])
    
    x_M[0:2] = R_inv @ x_M[0:2]

    x_M[0:2] = x_M[0:2] + np.array([-radius_2-cbf_radius,0])

    x_M[2] = wrap_to_pi(x[2] - p)

    return x_M

def eval_condition8(x):
    """
    Evaluate the condition for the first transformation.
    
    Args:
        x (array-like): The point in the domain.
        
    Returns:
        bool: True if the condition is satisfied, False otherwise.
    
    """

    condition_1 = np.dot(np.array([-1.0,1.0]),x[0:2] - point_3) >= 0 and np.dot(np.array([1.0,1.0]),x[0:2] - point_4) >= 0 and np.sqrt((x[0] - point_8[0])**2 + (x[1]-point_8[1])**2) < radius_2 + cbf_radius

    condition_2 = x[1] < point_8[1] + l2 and x[0] > point_8[0] - l2 and x[0] < point_8[0] + l2
    
    return condition_1 or condition_2

def p8(x):
    """
    Compute the parameter p that maps a given point with D on the set M.
    
    Args:
        x (array-like): The point in the domain.
        
    Returns:
        p (float): The parameter that maps the point with D on the set M, angle in rad.
    
    """
    
    x_tmp = np.array(x, dtype=float) - np.array([point_8[0],point_8[1],0])

    m_vec = np.array([1,0,0])

    dot_product = np.dot(m_vec[0:2], x_tmp[0:2])
    norm_m_vec = np.linalg.norm(m_vec[0:2])  
    norm_x = np.linalg.norm(x_tmp[0:2])

    cos_theta = dot_product / (norm_m_vec * norm_x)
    theta_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Clip to avoid NaN due to floating point errors

    # Check if the angle is in the correct range, otherwise ignore it
    if not (theta_rad <= 3/4*np.pi and theta_rad >= 1/4*np.pi):
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
    
    x_M = np.array(x, dtype=float) - np.array([point_8[0],point_8[1],0])

    R_inv = np.array([[np.cos(p), np.sin(p)],
                    [-np.sin(p), np.cos(p)]])
    
    x_M[0:2] = R_inv @ x_M[0:2]

    x_M[0:2] = x_M[0:2] + np.array([-radius_2-cbf_radius,0])

    x_M[2] = wrap_to_pi(x[2] - p)

    return x_M

def eval_condition9(x):
    """
    Evaluate the condition for the first transformation.
    
    Args:
        x (array-like): The point in the domain.
        
    Returns:
        bool: True if the condition is satisfied, False otherwise.
    
    """
    
    return np.sqrt(x[0]**2 + x[1]**2) < radius_0 + cbf_radius

def p9(x):
    """
    Compute the parameter p that maps a given point with D on the set M.
    
    Args:
        x (array-like): The point in the domain.
        
    Returns:
        p (float): The parameter that maps the point with D on the set M, angle in rad.
    
    """
    
    x_tmp = np.array(x, dtype=float)

    m_vec = np.array([1,0,0])

    dot_product = np.dot(m_vec[0:2], x_tmp[0:2])
    norm_m_vec = np.linalg.norm(m_vec[0:2])  
    norm_x = np.linalg.norm(x_tmp[0:2])

    cos_theta = dot_product / (norm_m_vec * norm_x)
    theta_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Clip to avoid NaN due to floating point errors

    # Adjust the angle based on the quadrant
    if x_tmp[1] < 0:
        theta_rad = -theta_rad

    return theta_rad

def D9(x, p):
    """
    Map points in the domain to points on the manifold M using the point dependent parameter p.
    
    Args:
        x (array-like): The point in the domain.
        p (float): The parameter that maps the point with D on the set M, angle in rad.
        
    Returns:
        x_M (array-like): The point on the manifold M.
    
    """
    
    x_M = np.array(x, dtype=float)

    R_inv = np.array([[np.cos(p), np.sin(p)],
                    [-np.sin(p), np.cos(p)]])
    
    x_M[0:2] = R_inv @ x_M[0:2]

    x_M[0:2] = x_M[0:2] + np.array([-radius_0-cbf_radius,0])

    x_M[2] = wrap_to_pi(x[2] - p)

    return x_M

def eval_condition10(x):
    """
    Evaluate the condition for the first transformation.
    
    Args:
        x (array-like): The point in the domain.
        
    Returns:
        bool: True if the condition is satisfied, False otherwise.
    
    """

    return h(x) >= 0.95*cbf_radius

def p10(x):
    """
    Compute the parameter p that maps a given point with D on the set M.
    
    Args:
        x (array-like): The point in the domain.
        
    Returns:
        p (float): The parameter that maps the point with D on the set M, angle in rad.
    
    """
    
    p = np.array([-2*turning_radius, 0.0, 0.0], dtype=float)

    return p

def D10(x, p):
    """
    Map points in the domain to points on the manifold M using the point dependent parameter p.
    
    Args:
        x (array-like): The point in the domain.
        p (float): The parameter that maps the point with D on the set M, angle in rad.
        
    Returns:
        x_M (array-like): The point on the manifold M.
    
    """
    
    return p



eval_conditions = [eval_condition1, eval_condition2, eval_condition3, eval_condition4,
                   eval_condition5, eval_condition6, eval_condition7, eval_condition8, 
                   eval_condition9, eval_condition10]
p_list = [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10]
D_list = [D1, D2, D3, D4, D5, D6, D7, D8, D9, D10]

########################################################################################
# 2. Synthesize the complete CBF module using the partially known CBF module and the transformations

print("CBF computation based on equivariances started...")

tic = time.time()
cbfModule_complete = multi_equi_cbf.multi_equi_cbf_synthesis(cbfModule_partially_known_cbf=cbfModule_inner_circle_partial,
                            D_list=D_list,
                            p_list=p_list,
                            eval_condtions=eval_conditions,
                            domain_lower_bound=domain_lower_bound_complete,
                            domain_upper_bound=domain_upper_bound_complete,
                            discretization=discretization_complete)
toc = time.time()
print("Extra computation time for the complete CBF: ", toc-tic, " seconds.")
print("Computation of complete CBF took in total ", toc-tic+cbfModule_inner_circle_partial.cbf.computation_time, " seconds.")

########################################################################################
# 3. Save the CBF module to a file
cbfModule_complete.cbf.partial_computation_time = cbfModule_inner_circle_partial.cbf.computation_time
cbfModule_complete.cbf.computation_time = toc-tic+cbfModule_inner_circle_partial.cbf.computation_time

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



