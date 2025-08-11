"""

    Computation of the CBF for the double integrator with a circular obstacle using EQUIVARIANCES.

    The CBF is paritally computed explicitly in a parallelized fashion, and the remaining part is computed using the equivariance of the bicycle kinematics.
    The results are saved in a json file.

    Adrian Wiltz, 2025

"""

if __name__ == "__main__":
    

    import sys
    import os

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
    sys.path.append(str('Predictive_CBF_synthesis_Toolbox'))

    import numpy as np
    from Dynamics.DoubleIntegrator import DoubleIntegrator
    from CBF.CBFmodule import CBFmodule
    import CBF.CBFcomputation as CBFcomputation
    from Equivariance_Module import equi_cbf
    from scipy.linalg import block_diag
    import casadi as ca
    import time

    ########################################################################################

    # some parameters
    num_of_batches_factor = 5      # determines the number of batches for parallel computation
    cbf_file_name = "d2_cbfm_equi.json"

    # create a dynamic system
    x0 = np.array([0,0,0,0])
    u_min = np.array([-1.0,-1.0])    # [minimum speed, minimum steering angle] in [m/s, rad/s]
    u_max = np.array([1.0,1.0])     # [maximum speed, maximum steering angle] in [m/s, rad/s]
    myDoubleIntegrator = DoubleIntegrator(x0=x0,
                    u_min=u_min,
                    u_max=u_max)

    #create a state constraint function

    def h(x):
        """State constraint function for both numpy and casadi type arguments"""

        v_min = np.array([-2,-2])
        v_max = np.array([2,2])

        xc = 0
        yc = 0
        r = 0

        factor = 100

        if isinstance(x[0], (np.ndarray, float, int)):
            h0 = np.sqrt((x[0]-xc)**2 + (x[1]-yc)**2) - r
            h1 = factor * np.minimum(v_max[0] - x[2], x[2] - v_min[0])
            h2 = factor * np.minimum(v_max[1] - x[3], x[3] - v_min[1])
        else:
            h0 = ca.sqrt((x[0]-xc)**2 + (x[1]-yc)**2) - r
            h1 = factor * ca.fmin(v_max[0] - x[2], x[2] - v_min[0])
            h2 = factor * ca.fmin(v_max[1] - x[3], x[3] - v_min[1])

        h = ca.fmin(ca.fmin(h0, h1), h2) if isinstance(x[0], ca.MX) else np.min([h0, h1, h2])

        return h0

    # create a terminal constraint function
    def cf(x):
        """Terminal constraint function for casadi type arguments"""

        xc = 0
        yc = 0
        r = 10
        h0 = lambda x: (x[0]-xc)**2 + (x[1]-yc)**2 - r**2
        h0_grad = lambda x: ca.vertcat(2*(x[0]-xc), 2*(x[1]-yc))
        orientation = x[2:]/ca.norm_2(x[2:])

        delta = 0

        return ca.vertcat(ca.dot(h0_grad(x), orientation),h0(x) - delta)

    # set parameters for the CBF module
    T = 12
    gamma = 1
    
    domain_lower_bound_partial = np.array([-20,0,-2.5,-2.5])
    domain_upper_bound_partial = np.array([0.1,0,2.5,2.5])
    discretization_partial = np.array([21,1,15,15])
    domain_lower_bound_complete = np.array([-14,-14,-2.5,-2.5])
    domain_upper_bound_complete = np.array([14,14,2.5,2.5])
    discretization_complete = np.array([29,29,15,15])

    print("Number of grid points to be computed: ", np.prod(discretization_partial))

    # create a CBF module
    myCBFmodule_partial = CBFmodule(h=h, 
                            dynamicSystem=myDoubleIntegrator, 
                            cf=cf, 
                            T=T, 
                            N=30,
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
    warmStartInputTrajectory_0 = np.array([u_min[0]*np.ones(myCBFmodule_partial.N), 
                                            np.zeros(myCBFmodule_partial.N)])   # max acceleration backward
    warmStartInputTrajectory_1 = np.array([u_max[0]*np.ones(myCBFmodule_partial.N),
                                            np.zeros(myCBFmodule_partial.N)])   # max accelreation forward
    warmStartInputTrajectory_2 = np.array([np.zeros(myCBFmodule_partial.N),
                                            u_min[1]*np.ones(myCBFmodule_partial.N)])   # max acceleration down
    warmStartInputTrajectories_3 = np.array([np.zeros(myCBFmodule_partial.N),
                                            u_max[1]*np.ones(myCBFmodule_partial.N)])   # max acceleration up

    warmStartInputTrajectories = np.array([warmStartInputTrajectory_0, warmStartInputTrajectory_1, warmStartInputTrajectory_2, warmStartInputTrajectories_3])

    myCBFmodule_partial.setWarmStartInputTrajectories(warmStartInputTrajectories)

    # 2. Compute the CBF on the domain
    tic = time.time()
    CBFcomputation.computeCbfParallelized(myCBFmodule_partial, processes=None, timeout_per_sample=10000, num_of_batches_factor=num_of_batches_factor)
    toc = time.time()
    computation_time_partial_cbf = toc-tic
    print("Computation of partially known CBF took ", computation_time_partial_cbf, " seconds.")

    # 3. Define p(x) and D(x)
    def p(x):
        """
        Compute the parameter p that maps a given point with D on the set M.
        
        Args:
            x (array-like): The point in the domain.
            
        Returns:
            p (float): The parameter that maps the point with D on the set M, angle in rad.
        
        """
        
        x_pos = x[0:2]
        
        m_vec = np.array([-1,0], dtype=float)

        dot_product = np.dot(m_vec, x_pos)
        norm_m_vec = np.linalg.norm(m_vec)  
        norm_x = np.linalg.norm(x_pos)

        cos_theta = dot_product / (norm_m_vec * norm_x)
        theta_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Clip to avoid NaN due to floating point errors

        # Adjust the angle based on the quadrant
        if x[1] > 0:
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
        
        R_inv_double_integrator = block_diag(R_inv, R_inv)

        x_M = R_inv_double_integrator @ x

        return x_M

    # 4. Compute the CBF for the entire domain
    tic = time.time()
    cbfModule_complete = equi_cbf.equi_cbf_synthesis(cbfModule_partially_known_cbf=myCBFmodule_partial,
                                D=D,
                                p=p,
                                domain_lower_bound=domain_lower_bound_complete,
                                domain_upper_bound=domain_upper_bound_complete,
                                discretization=discretization_complete)
    toc = time.time()
    print("Extra computation time for the complete CBF: ", toc-tic, " seconds.")
    print("Computation of complete CBF took in total ", toc-tic+computation_time_partial_cbf, " seconds.")

    # 3. Save the CBF module to a file
    cbfModule_complete.cbf.partial_computation_time = computation_time_partial_cbf
    cbfModule_complete.cbf.computation_time = toc-tic+computation_time_partial_cbf

    # 5. Save the CBF module of the complete CBF to a file
    cbfModule_complete.save(cbf_file_name, folder_name="Data")

    print("CBF computation finished.")



