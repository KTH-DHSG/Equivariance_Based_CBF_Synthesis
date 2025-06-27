"""

    Implementation of the linear system example from the CBF paper (Example 2 in Section IV-C).

    This script computes the CBF directly by computing all values of the CBF explicitly, which confirms the theoretically found symmetries of the CBF.

    Adrian Wiltz, 2025
    
"""

if __name__ == "__main__":
    ########################################################################################
    import sys
    import os

    target_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Predictive_CBF_synthesis_Toolbox'))

    sys.path.append(target_path)

    import numpy as np
    from CBF.CBFmodule import CBFmodule
    import CBF.CBFcomputation as CBFcomputation
    from Dynamics.LinearSystem import LinearSystem
    import casadi as ca
    import matplotlib.pyplot as plt
    import time

    ########################################################################################

    # some parameters
    num_of_batches_factor = 10      # determines the number of batches for parallel computation
    cbf_file_name = "linear_system_1.json"

    # create a dynamic system
    t0 = 0
    x0 = np.array([0,0])
    u_min = np.array([-np.inf])    
    u_max = np.array([np.inf])     
    myLinearSystem = LinearSystem(
                    A=np.array([[1,2],[3,-4]]),
                    B=np.array([[-1],[3]]),
                    x0=x0,
                    u_min=u_min,
                    u_max=u_max)

    #create a state constraint function

    def h(x):
        """State constraint function"""
        w_i = np.array([-1,2])
        offset = -2

        h = w_i[0]*x[0] + w_i[1]*x[1] + offset

        return h

    # create a terminal constraint function
    def cf(x):
        """Terminal constraint function for casadi type arguments"""

        w_i = np.array([-1,2])
        offset = -4

        cf = w_i[0]*x[0] + w_i[1]*x[1] + offset

        return cf

    # set parameters for the CBF module
    T = 1
    gamma = -0.01

    # set domain bounds
    domain_lower_bound = np.array([-4,-4])
    domain_upper_bound = np.array([4,4])
    discretization = np.array([40,40])

    print("Number of grid points to be computed without exploiting symmetries: ", np.prod(discretization))

    # create a CBF module
    myCBFmodule = CBFmodule(h=h, 
                            dynamicSystem=myLinearSystem, 
                            cf=cf, 
                            T=T, 
                            N=40,
                            gamma=gamma, 
                            domain_lower_bound=domain_lower_bound, 
                            domain_upper_bound=domain_upper_bound, 
                            discretization=discretization,
                            p_norm=50,
                            p_norm_decrement=10,
                            p_norm_min=40)

    ########################################################################################
    # Initialize the cbf value optimization and compute the cbf value at a selction of sample points

    # 1. Initialize the warm start input trajectories and assign them to the cbf module
    warmStartInputTrajectory_1 = 0*np.ones((1,myCBFmodule.N))
    warmStartInputTrajectory_2 = 50*np.ones((1,myCBFmodule.N))

    warmStartInputTrajectories = np.array([warmStartInputTrajectory_1])
    myCBFmodule.setWarmStartInputTrajectories(warmStartInputTrajectories)

    # 2. Compute the CBF on the domain
    tic = time.time()
    CBFcomputation.computeCbfParallelized(myCBFmodule, processes=None, timeout_per_sample=20, num_of_batches_factor=num_of_batches_factor)
    toc = time.time()
    print("CBF computation took ", toc-tic, " seconds.")

    # 3. Save the CBF module to a file
    myCBFmodule.cbf.computation_time = toc-tic

    # 4. Save the CBF module to a file
    myCBFmodule.save(cbf_file_name, folder_name="Data")

    print("CBF computation finished.")

