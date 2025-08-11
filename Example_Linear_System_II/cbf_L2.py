"""

    Implementation of the example for the linear system with rotational equivariances from the CBF paper.

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
    from LinearSystem_L2 import LinearSystem_L2
    from Dynamics.DynamicSystem import DynamicSystem
    import casadi as ca
    import matplotlib.pyplot as plt
    import time

    ########################################################################################

    # some parameters
    num_of_batches_factor = 20      # determines the number of batches for parallel computation
    cbf_file_name = "cbf_linear_system_2.json"

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

    # set domain bounds
    domain_lower_bound = np.array([-3,-3])
    domain_upper_bound = np.array([3,3])
    discretization = np.array([40,40])

    print("Number of grid points to be computed without exploiting symmetries: ", np.prod(discretization))

    # create a CBF module
    myCBFmodule = CBFmodule(h=h, 
                            dynamicSystem=myLinearSystem, 
                            cf=cf, 
                            T=T, 
                            N=20,
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
    warmStartInputTrajectory_1 = np.array([np.zeros(myCBFmodule.N),
                                        3*np.ones(myCBFmodule.N)])
    warmStartInputTrajectory_2 = np.array([np.zeros(myCBFmodule.N),
                                        -3*np.ones(myCBFmodule.N)])
    warmStartInputTrajectory_3 = np.array([np.pi/2*np.ones(myCBFmodule.N),
                                        3*np.ones(myCBFmodule.N)])
    warmStartInputTrajectory_4 = np.array([-np.pi/2*np.ones(myCBFmodule.N),
                                        3*np.ones(myCBFmodule.N)])

    warmStartInputTrajectories = np.array([warmStartInputTrajectory_1,
                                        warmStartInputTrajectory_2,
                                        warmStartInputTrajectory_3,
                                        warmStartInputTrajectory_4])
    myCBFmodule.setWarmStartInputTrajectories(warmStartInputTrajectories)

    # 2. Compute the CBF on the domain
    tic = time.time()
    CBFcomputation.computeCbfParallelized(myCBFmodule, processes=None, timeout_per_sample=200, num_of_batches_factor=num_of_batches_factor)
    toc = time.time()
    print("CBF computation took ", toc-tic, " seconds.")

    # 3. Save the CBF module to a file
    myCBFmodule.cbf.computation_time = toc-tic

    # 4. Save the CBF module to a file
    myCBFmodule.save(cbf_file_name, folder_name="Data")

    print("CBF computation finished.")

