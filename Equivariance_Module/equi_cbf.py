"""

    This module provides functions to compute the complete Control Barrier Function (CBF) using the equivariance property of the CBF. It also allows for parallelized computation using Dask. It is important to note that the parallelized computation is only beneficial for very large batches due to the overhead of parallelization.

    The functions require the knowledge of an at least partially known CBF, and a transformation, that shifts the known CBF along the boundary of the constraint set.

    The main functions in this module are:
    - equi_cbf_synthesis: Computes the complete CBF for a given partially known
        CBF by using the equivariance property of the CBF.
    - equi_cbf_synthesis_from_saved_cbf: Computes the complete CBF from a saved
        partially known CBF file.
    - equi_cbf_synthesis_parallelized: Computes the complete CBF in parallel using Dask.
    - __equi_cbf_synthesis_parallelized_worker__: A worker function for the parallel
        computation of the CBF values for a given batch of points.

    Adrian Wiltz, 2025

"""

import copy
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import numpy as np
from CBF.CBFmodule import CBFmodule
from tqdm import tqdm  
from dask.distributed import Client, as_completed       # dask client
import webbrowser
import asyncio
from tqdm import tqdm

def equi_cbf_synthesis(
        cbfModule_partially_known_cbf,
        D,
        p,
        domain_lower_bound,
        domain_upper_bound,
        discretization
    ):
    """This function computes the complete CBF for a given partially known CBF by using the equivariance property of the CBF. It uses the precomputed CBF values and interpolates them to fill in the missing values.
    
    The function takes the following parameters:
    
    Parameters:
        cbfModule_partially_known_cbf (CBFmodule): The cbf module for the partially known CBF.
        D (function): A function that maps points in the domain to points on the manifold M using the point dependent parameter p.
        p (function): A function that computes the parameter that maps a given point with D on the set M. The return value of p can be either a scalar, or an iterable of any data type.
        domain_lower_bound (array-like): The lower bound of the domain.
        domain_upper_bound (array-like): The upper bound of the domain.
        discretization (array-like): The discretization of the domain.
        
    Returns:
        cbfModule_complete (CBFmodule): The complete CBF module with the computed CBF values for the entire domain.
    
    """

    # Load the precomputed, partially known CBF
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
                        p_norm_min=cbfModule_partial.p_norm_min
    )

    complete_point_list = cbfModule_complete.cbf.getPointList()

    cbf_interpolator = cbfModule_partial.cbf.getCbfInterpolator()

    for point_element in tqdm(complete_point_list, desc="Computing CBF values"):
        # Compute the CBF value for each point in the domain
        current_point = point_element["point"]
        index = point_element["index"]
        params = p(current_point)
        params = np.atleast_1d(params)  # Ensure params is an array
        cbf_value_candidates = np.zeros(len(params))
        for k in range(len(params)):
            param_tmp = params[k]
            current_point_on_M = D(current_point, param_tmp)
            cbf_value_candidates[k] = cbf_interpolator(current_point_on_M)
            
        cbfModule_complete.cbf.cbf_values[index] = np.nanmax(cbf_value_candidates)

        if current_point[2] < np.pi/2+0.1 and current_point[2] > np.pi/2-0.1:
            pass

    return cbfModule_complete

def equi_cbf_synthesis_from_saved_cbf(
        cbf_file_name,
        cbf_folder_name,
        D,
        p,
        domain_lower_bound,
        domain_upper_bound,
        discretization
    ):
    """
    This function computes the complete CBF for a given partially known CBF by using the equivariance property of the CBF. It uses the precomputed CBF values and interpolates them to fill in the missing values.
    The function takes the following parameters:

    Parameters:
        cbf_file_name (str): The name of the file containing the precomputed CBF.
        cbf_folder_name (str): The folder where the CBF file is located.
        D (function): A function that maps points in the domain to points on the manifold M using the point dependent parameter p.
        p (function): A function that computes the parameter that maps a given point with D on the set M.
        domain_lower_bound (array-like): The lower bound of the domain.
        domain_upper_bound (array-like): The upper bound of the domain.
        discretization (array-like): The discretization of the domain.

    Returns:
        cbfModule_complete (CBFmodule): The complete CBF module with the computed CBF values for the entire domain.

    """

    # Load the precomputed, partially known CBF
    cbfModule_partial = CBFmodule()
    cbfModule_partial.load(filename=cbf_file_name, folder_name=cbf_folder_name)

    cbfModule_complete = equi_cbf_synthesis(
                            cbfModule_partial,
                            D,
                            p,
                            domain_lower_bound,
                            domain_upper_bound,
                            discretization
                        )
    
    return cbfModule_complete

def equi_cbf_synthesis_parallelized(
        cbf_file_name,
        cbf_folder_name,
        D,
        p,
        domain_lower_bound,
        domain_upper_bound,
        discretization,
        cbfModule_partial=None,
        num_of_batches_factor=4,
        processes=None,
        timeout_per_sample=20
    ):
    """
    This function computes the complete CBF for a given partially known CBF by using the equivariance property of the CBF. It uses the precomputed CBF values and interpolates them to fill in the missing values. The CBF is computed in parallel using Dask.
    
    IMPORTANT REMARK: Only for very large batches a computational advantage is achieved due to the parallelization overhead.
    
    Parameters:
        cbf_file_name (str): The name of the file containing the precomputed CBF.
        cbf_folder_name (str): The folder where the CBF file is located.
        D (picklable function): A function that maps points in the domain to points on the manifold M using the point dependent parameter p. To be picklable, the function should not be a lambda function or a nested function.
        p (picklablefunction): A function that computes the parameter that maps a given point with D on the set M. To be picklable, the function should not be a lambda function or a nested function.
        domain_lower_bound (array-like): The lower bound of the domain.
        domain_upper_bound (array-like): The upper bound of the domain.
        discretization (array-like): The discretization of the domain.
        cbfModule_partial (CBFmodule, optional): The cbf module for the partially known CBF. If None, the function will load the CBF from the file. Default is None.
        num_of_batches_factor (int, optional): The factor by which to multiply the number of processes to determine the number of batches. Default is 4.
        processes (int, optional): The number of processes to use for parallel computation. If None, uses os.cpu_count(). Default is None.
        timeout_per_sample (int, optional): The timeout for each sample in seconds. Default is 20.
        
    Returns:
        None: The function does not return anything. It modifies the CBF module in place.
    
    """

    # Load the precomputed, partially known CBF
    if cbfModule_partial is None:
        cbfModule_partial = CBFmodule()
        cbfModule_partial.load(filename=cbf_file_name, folder_name=cbf_folder_name)

    # Create cbf module for the complete CBF, still to be computed
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

    # Get the list of points in the complete CBF
    complete_point_list = cbfModule_complete.cbf.getPointList()

    cbf_values = [-np.inf] * len(complete_point_list)
    for i, point in enumerate(complete_point_list):
        point["cbf_value"] = cbf_values[i]

    # Determine the number of batches and processes
    if processes is None:
        processes = os.cpu_count()

    num_of_batches = processes * num_of_batches_factor

    # Split the list of points into batches
    batch_size = len(complete_point_list)//num_of_batches
    batches = [complete_point_list[i:i + batch_size] for i in range(0, len(complete_point_list), batch_size)]

    # Create a list of the other parameters
    cbf_file_name_list = [cbf_file_name for _ in range(len(batches))]
    cbf_folder_name_list = [cbf_folder_name for _ in range(len(batches))]
    D_list = [D for _ in range(len(batches))]
    p_list = [p for _ in range(len(batches))]

    # Initialize the dask client
    timeout = timeout_per_sample * batch_size
    client = Client(processes=True, n_workers=processes, threads_per_worker=1, memory_limit='2GB', death_timeout=180)
    webbrowser.open(client.dashboard_link)
    futures = client.map(__equi_cbf_synthesis_parallelized_worker_sync__, cbf_file_name_list, cbf_folder_name_list, batches, D_list, p_list, retries=3)

    for future in tqdm(as_completed(futures, timeout=timeout), total=len(futures), desc="Computing CBF via equivariances [batches computed/total batches]"):
        result = future.result()
        for batch_element in result:
            index = batch_element["index"]
            cbf_value = batch_element["cbf_value"]
            cbfModule_complete.cbf.cbf_values[index] = cbf_value

    # Close the dask client
    client.close()
    
    print("CBF computation via equivariances completed.")

    return cbfModule_complete

async def __equi_cbf_synthesis_parallelized_worker__(
        cbf_file_name,
        cbf_folder_name,
        batch,
        D,
        p
    ):
    """
    This function computes the CBF values for a given batch of points using the equivariance property of the CBF. It uses the precomputed CBF values and interpolates them to fill in the missing values. The function is designed to be used in a parallelized context, such as with Dask. It is called by the `equi_cbf_synthesis_parallelized` function.
    
    The function takes the following parameters:
    
    Parameters:
        cbf_file_name (str): The name of the file containing the precomputed CBF.
        cbf_folder_name (str): The folder where the CBF file is located.
        batch (list): A list of points for which to compute the CBF values.
        D (function): A function that maps points in the domain to points on the manifold M using the point dependent parameter p.
        p (function): A function that computes the parameter that maps a given point with D on the set M.
        
    Returns:
        batch (list): The input batch with the computed CBF values for each point.
    
    """

    batch = copy.deepcopy(batch)

    # Load the precomputed, partially known CBF
    cbfModule_partial = CBFmodule()
    cbfModule_partial.load(filename=cbf_file_name, folder_name=cbf_folder_name)

    cbf_interpolator = cbfModule_partial.cbf.getCbfInterpolator()

    for point_element in batch:
        # Compute the CBF value for each point in the domain
        current_point = point_element["point"]
        index = point_element["index"]
        param_in_current_point = p(current_point)
        current_point_on_M = D(current_point, param_in_current_point)
        point_element["cbf_value"] = cbf_interpolator(current_point_on_M)

    return batch

def __equi_cbf_synthesis_parallelized_worker_sync__(*args):
    return asyncio.run(__equi_cbf_synthesis_parallelized_worker__(*args))