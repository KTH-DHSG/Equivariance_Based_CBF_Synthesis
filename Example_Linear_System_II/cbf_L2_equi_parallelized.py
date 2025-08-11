"""

    Implementation of the example for the linear system with rotational equivariances from the CBF paper.

    This script computes the the CBF based on equivariances using parallelization.
    Note: Due to the parallelization overhead and the small number of points to be computed, the computation time is longer that for the non-parallelized version.

    Adrian Wiltz, 2025
    
"""

if __name__ == "__main__":
    ########################################################################################
    import sys
    import os

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    sys.path.append(str('Predictive_CBF_synthesis_Toolbox'))

    import numpy as np
    from CBF.CBFmodule import CBFmodule
    import CBF.CBFcomputation as CBFcomputation
    from LinearSystem_L2 import LinearSystem_L2
    from Dynamics.DynamicSystem import DynamicSystem
    from Equivariance_Module import equi_cbf
    import casadi as ca
    import matplotlib.pyplot as plt
    import time

    ########################################################################################

    # some parameters
    num_of_batches_factor = 4      # determines the number of batches for parallel computation

    cbf_partial_file_name = "cbf_linear_system_2_partial.json"
    cbf_partial_folder_name = r'Data'

    cbf_file_name = "cbf_linear_system_2.json"
    cbf_folder_name = r'Example_Linear_System_II/Data'

    # create a dynamic system
    t0 = 0
    x0 = np.array([0,0])
    u_min = np.array([-np.pi/2,-3])    
    u_max = np.array([np.pi/2,3])     
    myLinearSystem = LinearSystem_L2(
                    A=np.array([[-1,-2],[2,-1]]),
                    B=np.array([[1,0],[0,1]]),
                    x0=x0,
                    u_min=u_min,
                    u_max=u_max)

    #create a state constraint function

    def h(x):
        """State constraint function"""

        return np.sqrt(x[0]**2 + x[1]**2) - 1

    # create a terminal constraint function
    def cf(x):
        """Terminal constraint function for casadi type arguments"""

        return np.sqrt(x[0]**2 + x[1]**2) - 2

    # set parameters for the CBF module
    T = 2
    gamma = -0.01

    # set domain bounds for partially computed CBF
    domain_lower_bound_partial = np.array([-0.1,-4.5])
    domain_upper_bound_partial = np.array([0.1,0.1])
    discretization_partial = np.array([3,40])

    # set domain bounds for partially computed CBF
    domain_lower_bound_complete = np.array([-3,-3])
    domain_upper_bound_complete = np.array([3,3])
    discretization_complete = np.array([40,40])

    print("Number of grid points computed for the partially known CBF: ", np.prod(discretization_partial))

    # create a CBF module
    myCBFmodule_partial = CBFmodule(h=h, 
                            dynamicSystem=myLinearSystem, 
                            cf=cf, 
                            T=T, 
                            N=20,
                            gamma=gamma, 
                            domain_lower_bound=domain_lower_bound_partial, 
                            domain_upper_bound=domain_upper_bound_partial, 
                            discretization=discretization_partial,
                            p_norm=50,
                            p_norm_decrement=10,
                            p_norm_min=40)

    ########################################################################################
    # Initialize the cbf value optimization and compute the cbf value at a selction of sample points

    # 1. Initialize the warm start input trajectories and assign them to the cbf module
    warmStartInputTrajectory_1 = np.array([np.zeros(myCBFmodule_partial.N),
                                        3*np.ones(myCBFmodule_partial.N)])

    warmStartInputTrajectories = np.array([warmStartInputTrajectory_1])
    myCBFmodule_partial.setWarmStartInputTrajectories(warmStartInputTrajectories)

    # 2. Compute the CBF on the domain
    tic = time.time()
    CBFcomputation.computeCbfParallelized(myCBFmodule_partial, processes=None, timeout_per_sample=200, num_of_batches_factor=num_of_batches_factor)
    toc = time.time()
    computation_time_partial_cbf = toc-tic
    print("Computation of partially known CBF took ", computation_time_partial_cbf, " seconds.")

    # IMPORTANT: Update the file name variable!
    cbf_partial_file_name = myCBFmodule_partial.save(cbf_partial_file_name, cbf_partial_folder_name)

    # 3. Define p(x) and D(x)
    def p(x):
        """
        Compute the parameter p that maps a given point with D on the set M.
        
        Args:
            x (array-like): The point in the domain.
            
        Returns:
            p (float): The parameter that maps the point with D on the set M, angle in rad.
        
        """
        
        m_vec = np.array([0,-1])

        dot_product = np.dot(m_vec, x)
        norm_m_vec = np.linalg.norm(m_vec)  
        norm_x = np.linalg.norm(x)

        cos_theta = dot_product / (norm_m_vec * norm_x)
        theta_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Clip to avoid NaN due to floating point errors

        # Adjust the angle based on the quadrant
        if x[0] < 0:
            theta_rad = -theta_rad

        return theta_rad
    
    def D(x, p):
        """
        Map points in the domain to points on the manifold M using the point dependent parameter p.
        
        Args:
            x (array-like): The point in the domain.
            p (float): The parameter that maps the point with D on the set M, angle in rad.
            
        Returns:
            x_M (array-like): The point on the manifold M.
        
        """

        R_inv = np.array([[np.cos(p), np.sin(p)],
                      [-np.sin(p), np.cos(p)]])
        
        x_M = R_inv @ x

        return x_M
        

    # 4. Compute the CBF for the entire domain
    tic = time.time()
    cbfModule_complete = equi_cbf.equi_cbf_synthesis_parallelized(
                                cbf_file_name=cbf_partial_file_name,
                                cbf_folder_name=cbf_partial_folder_name,
                                D=D,
                                p=p,
                                domain_lower_bound=domain_lower_bound_complete,
                                domain_upper_bound=domain_upper_bound_complete,
                                discretization=discretization_complete,
                                cbfModule_partial=myCBFmodule_partial,
                                num_of_batches_factor=1,
                                processes=4,
                                timeout_per_sample=20
                                )
    toc = time.time()
    print("Extra computation time for the complete CBF: ", toc-tic, " seconds.")
    print("Computation of complete CBF took in total ", toc-tic+computation_time_partial_cbf, " seconds.")
    
    # 5. Save the CBF module of the complete CBF to a file
    cbfModule_complete.save(cbf_file_name, folder_name="Data")

    print("CBF computation finished.")

