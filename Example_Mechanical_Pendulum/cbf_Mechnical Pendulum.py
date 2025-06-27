"""

    Computation of the CBF for the mechanical pendulum from the example in Section IV-A of the CBF paper.

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
    from MechanicalPendulum import MechnicalPendulum
    import casadi as ca
    import matplotlib.pyplot as plt
    import time

    ########################################################################################

    # some parameters
    num_of_batches_factor = 10      # determines the number of batches for parallel computation
    cbf_file_name = "cbf_mechanical_pendulum.json"

    # create a dynamic system
    t0 = 0
    x0 = np.array([1.3,-1.8])
    u_min = np.array([-5])    
    u_max = np.array([5])     
    myPendulum = MechnicalPendulum(
                    L=1,
                    x0=x0,
                    u_min=u_min,
                    u_max=u_max)

    #create a state constraint function

    def h(x):
        """State constraint function"""

        a = 1
        b = 2

        h = -np.sqrt((x[0]+x[1])**2/(2 * a**2) + (-x[0]+x[1])**2/(2 * b**2)) + 1

        return h

    # create a terminal constraint function
    def cf(x):
        """Terminal constraint function for casadi type arguments"""

        cf = -np.sqrt(2 * x[0]**2 + x[1]**2 + 2*x[0]*x[1]) + 0.5
        
        return cf

    # set parameters for the CBF module
    T = 2
    gamma = -0.01

    # set domain bounds
    domain_lower_bound = np.array([-2,-2])
    domain_upper_bound = np.array([2,2])
    discretization = np.array([40,40])

    print("Number of grid points to be computed without exploiting symmetries: ", np.prod(discretization))

    # create a CBF module
    myCBFmodule = CBFmodule(h=h, 
                            dynamicSystem=myPendulum, 
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
    warmStartInputTrajectory_0 = np.zeros((1,myCBFmodule.N))

    warmStartInputTrajectories = np.array([warmStartInputTrajectory_0])
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

