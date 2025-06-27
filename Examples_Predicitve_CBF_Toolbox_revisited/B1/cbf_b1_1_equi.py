"""

    Computation of the CBF for the less agile bicycle model with a circular obstacle using EQUIVARIANCES.

    The CBF is partitally computed explicitly in a parallelized fashion, and the remaining part is computed using the equivariance of the bicycle kinematics.
    The results are saved in a json file.

    Adrian Wiltz, 2025
    
"""

if __name__ == '__main__':

    import sys
    import os

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

    import numpy as np
    from Dynamics.Bicycle import Bicycle
    from CBF.CBFmodule import CBFmodule
    import CBF.CBFcomputation as CBFcomputation
    from Equivariance_Module import equi_cbf
    from math_aux.math_aux import wrap_to_pi
    import casadi as ca
    import time

    ########################################################################################
    # Specify the system dynamics, the CBF computation parameters, and initialize the CBF module
    
    # some parameters
    num_of_batches_factor = 5      # determines the number of batches for parallel computation
    cbf_file_name = "b1_1_cbfm_2p8_equi_reduced.json"
    cbf_file_name_partial = "b1_1_cbfm_2p8_partial_equi_reduced.json"
    
    # create a dynamic system
    t0 = 0
    x0 = np.array([0,0,0])
    L = 1
    u_min = np.array([1, -20/180*np.pi])    # [minimum speed, minimum steering angle] in [m/s, rad/s]
    u_max = np.array([2, 20/180*np.pi])     # [maximum speed, maximum steering angle] in [m/s, rad/s]
    myBike = Bicycle(x0=x0,
                    L=1,
                    u_min=u_min,
                    u_max=u_max)

    #create a state constraint function

    def h(x):
        """State constraint function"""
        xc = 0
        yc = 0
        r = 0
        return np.sqrt((x[0]-xc)**2 + (x[1]-yc)**2) - r

    # create a terminal constraint function
    def cf(x):
        """Terminal constraint function for casadi type arguments"""

        xc = 0
        yc = 0
        r = 15
        h = lambda x: (x[0]-xc)**2 + (x[1]-yc)**2 - r**2
        turning_radius = 2.8
        h_grad = lambda x: ca.vertcat(2*(x[0]-xc), 2*(x[1]-yc))
        delta = (2*turning_radius)**2
        orientation = ca.vertcat(ca.cos(x[2]), ca.sin(x[2]))

        return ca.vertcat(ca.dot(h_grad(x), orientation), h(x) - delta)

    # set parameters for the CBF module
    T = 12
    gamma = 2

    domain_lower_bound_partial = np.array([-21.3,0,-np.pi])
    domain_upper_bound_partial = np.array([0.1,0,0.0])
    discretization_partial = np.array([43,1,21])
    domain_lower_bound_partial_2 = np.array([-21.3,0,-np.pi])
    domain_upper_bound_partial_2 = np.array([0.1,0,np.pi])
    discretization_partial_2 = np.array([43,1,41])
    domain_lower_bound_complete = np.array([-15,-15,-np.pi])
    domain_upper_bound_complete = np.array([15,15,np.pi])
    discretization_complete = np.array([61,61,41])

    print("Number of grid points to be computed: ", np.prod(discretization_partial))

    # create a CBF module
    myCBFmodule_partial = CBFmodule(h=h, 
                            dynamicSystem=myBike, 
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

    curve_steps = 10

    # 1. Initialize the warm start input trajectories and assign them to the cbf module
    warmStartInputTrajectory_0 = np.array([u_max[0]*np.ones(myCBFmodule_partial.N), 
                                            u_min[1]*np.ones(myCBFmodule_partial.N)])   # max speed and turn to the right
    warmStartInputTrajectory_0[1,curve_steps:] = np.zeros(myCBFmodule_partial.N-curve_steps) # set the steering angle to zero after curve
    warmStartInputTrajectory_1 = np.array([u_max[0]*np.ones(myCBFmodule_partial.N),
                                            np.zeros(myCBFmodule_partial.N)])   # go straight at max speed
    warmStartInputTrajectory_2 = np.array([u_max[0]*np.ones(myCBFmodule_partial.N),
                                            u_max[1]*np.ones(myCBFmodule_partial.N)])   # max speed and turn to the left
    warmStartInputTrajectory_2[1,curve_steps:] = np.zeros(myCBFmodule_partial.N-curve_steps) # set the steering angle to zero after curve
    warmStartInputTrajectories = np.array([warmStartInputTrajectory_0, warmStartInputTrajectory_1, warmStartInputTrajectory_2])


    myCBFmodule_partial.setWarmStartInputTrajectories(warmStartInputTrajectories)

    # 2. Compute the CBF on the domain
    tic = time.time()
    CBFcomputation.computeCbfParallelized(myCBFmodule_partial, processes=None, timeout_per_sample=300, num_of_batches_factor=num_of_batches_factor)
    toc = time.time()
    computation_time_partial_cbf = toc-tic
    print("Computation of partially known CBF took ", computation_time_partial_cbf, " seconds.")

    # myCBFmodule_partial.save(cbf_file_name_partial, folder_name="Data")

    # 3. Use orientational equivariance of the bicycle kinematics
    def p1(x):
        """
        Compute the parameter p that maps a given point with D on the set M.
        
        Args:
            x (array-like): The point in the domain.
            
        Returns:
            p (float): The parameter that maps the point with D on the set M, angle in rad.
        
        """

        return -np.abs(x[2])
    
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
        
        if x_M[2] != p:
            # If angle if mirrored, flip also the y-coordinate
            x_M[1] = -x_M[1]

        x_M[2] = p

        return x_M
    
    tic = time.time()
    cbfModule_partial_2 = equi_cbf.equi_cbf_synthesis(cbfModule_partially_known_cbf=myCBFmodule_partial,
                                D=D1,
                                p=p1,
                                domain_lower_bound=domain_lower_bound_partial_2,
                                domain_upper_bound=domain_upper_bound_partial_2,
                                discretization=discretization_partial_2)
    toc = time.time()
    print("Extra computation time for the CBF_partial_2: ", toc-tic, " seconds.")
    print("Computation of CBF_partial_2 took in total ", toc-tic+computation_time_partial_cbf, " seconds.")

    computation_time_partial_cbf_2 = toc-tic+computation_time_partial_cbf

    # 4. Exploit the rotational equivariance of the bicycle kinematics
    def p2(x):
        """
        Compute the parameter p that maps a given point with D on the set M.
        
        Args:
            x (array-like): The point in the domain.
            
        Returns:
            p (float): The parameter that maps the point with D on the set M, angle in rad.
        
        """
        
        m_vec = np.array([-1,0,0], dtype=float)

        dot_product = np.dot(m_vec[0:2], x[0:2])
        norm_m_vec = np.linalg.norm(m_vec[0:2])  
        norm_x = np.linalg.norm(x[0:2])

        cos_theta = dot_product / (norm_m_vec * norm_x)
        theta_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Clip to avoid NaN due to floating point errors

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
        
        x_M = np.array(x, dtype=float)

        R_inv = np.array([[np.cos(p), np.sin(p)],
                      [-np.sin(p), np.cos(p)]])
        
        x_M[0:2] = R_inv @ x[0:2]

        x_M[2] = wrap_to_pi(x[2] - p)

        return x_M

    tic = time.time()
    cbfModule_complete = equi_cbf.equi_cbf_synthesis(cbfModule_partially_known_cbf=cbfModule_partial_2,
                                D=D2,
                                p=p2,
                                domain_lower_bound=domain_lower_bound_complete,
                                domain_upper_bound=domain_upper_bound_complete,
                                discretization=discretization_complete)
    toc = time.time()
    print("Extra computation time for the complete CBF: ", toc-tic, " seconds.")
    print("Computation of complete CBF took in total ", toc-tic+computation_time_partial_cbf_2, " seconds.")

    # 3. Save the CBF module to a file
    cbfModule_complete.cbf.partial_computation_time = computation_time_partial_cbf
    cbfModule_complete.cbf.computation_time = toc-tic+computation_time_partial_cbf_2

    # 5. Save the CBF module of the complete CBF to a file
    cbfModule_complete.save(cbf_file_name, folder_name="Data")

    print("CBF computation finished.")



