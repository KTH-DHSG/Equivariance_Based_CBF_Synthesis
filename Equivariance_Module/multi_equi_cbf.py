"""

    This module provides functions to compute the complete Control Barrier Function (CBF) using the equivariance property of CBFs. 
    The functions in this module is similar to that in equi_cbf.py, with the difference that it is designed for constraints that require multiple transformations and parameter functions for the CBF synthesis.

    The functions require the knowledge of an at least partially known CBF, and a transformation, that shifts the known CBF along the boundary of the constraint set.

    The main functions in this module are:
        - multi_equi_cbf_synthesis: Synthesizes a complete CBF module using multiple transformations and parameter functions.
        
    Adrian Wiltz, 2025
    
"""

import copy
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
sys.path.append(str('Predictive_CBF_synthesis_Toolbox'))

import numpy as np
from CBF.CBFmodule import CBFmodule
from tqdm import tqdm  

def multi_equi_cbf_synthesis(
        cbfModule_partially_known_cbf,
        D_list,
        p_list,
        eval_condtions,
        domain_lower_bound,
        domain_upper_bound,
        discretization
    ):
    """
    Synthesize a complete CBF module using multiple transformations and parameter functions. The function eval conditions returns ture, if a particular transformation shall be applied to the point in the domain. The CBF value equals the maximum value of all the transformed points. If none of the evaluation conditions is satisfied for a particular point, then it is assumed that the point is sufficiently remote from any boundary and np.inf is assigned to the CBF value.

    Args:
        cbfModule_partially_known_cbf (CBFmodule): The partially known CBF provided as CBF module.
        D_list (list): List of transformation functions.
        p_list (list): List of parameter functions.
        eval_condtions (list): List of evaluation condition functions.
        domain_lower_bound (array-like): Lower bound of the domain of the CBF to be determined.
        domain_upper_bound (array-like): Upper bound of the domain of the CBF to be determined.
        discretization (array-like): Discretization of the domain of the CBF to be determined.

    Returns:
        CBFmodule: The complete CBF module.

    """

    if len(D_list) != len(p_list) and len(D_list) != len(eval_condtions):
        raise ValueError("D_list, p_list and eval_conditions must have the same length.")

    cbfModule_partial = cbfModule_partially_known_cbf

    cbfModule_complete = CBFmodule(
                        h=cbfModule_partial.h,
                        dynamicSystem=cbfModule_partial.dynamics,
                        cf=cbfModule_partial.terminal_condition,
                        T=cbfModule_partial.T,
                        N=cbfModule_partial.N,
                        gamma=cbfModule_partial.gamma,
                        domain_lower_bound=domain_lower_bound,
                        domain_upper_bound=domain_upper_bound,
                        discretization=discretization,
                        p_norm=cbfModule_partial.p_norm,
                        p_norm_decrement=cbfModule_partial.p_norm_decrement,
                        p_norm_min=cbfModule_partial.p_norm_min,
                    )

    complete_point_list = cbfModule_complete.cbf.getPointList()

    cbf_interpolator = cbfModule_partial.cbf.getCbfInterpolator()

    num_transformations = len(D_list)

    for point_element in tqdm(complete_point_list, desc="Computing CBF values"):

        cbf_vals_tmp = -np.inf*np.ones(num_transformations)
        current_point = point_element["point"]
        index = point_element["index"]

        # For each point in the domain, check all provided transformations and determine the corresponding CBF value if the respective evaluation condition is satisfied
        for i in range(num_transformations):
            
            if eval_condtions[i](current_point):
                # Compute the transformed point
                current_param = p_list[i](current_point)
                current_point_on_M = D_list[i](current_point, current_param)

                # Compute the CBF value for the transformed point
                cbf_vals_tmp[i] = cbf_interpolator(current_point_on_M)
        
        # Determine the maximum CBF value for the current point
        cbf_vals_max = np.nanmax(cbf_vals_tmp)

        if cbf_vals_max > -np.inf:
            # Assign the maximum CBF value to the corresponding index in the complete CBF module
            cbfModule_complete.cbf.cbf_values[index] = cbf_vals_max
        else:
            # If no valid CBF value was found, assume that the point is sufficiently remote from any boundary and assign np.inf to the CBF value
            cbfModule_complete.cbf.cbf_values[index] = np.inf

    return cbfModule_complete


    