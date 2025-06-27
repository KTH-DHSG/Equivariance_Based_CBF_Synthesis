"""

    Equivariance based CBF synthesis for the bicycle model using the knowledge over a partially known CBF. 
    The constraint under consideration is a 

    >> non-convex circular obstacle (feasible set is circular)

    Adrian Wiltz, 2025

"""

if __name__ == '__main__':

    import sys
    import os

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

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
    cbf_file_name = "cbf_inner_circle_complete.json"
    
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
    
    # turning radius of the bicycle assuming that u_max[1] = -u_min[1]
    beta_max = np.arctan(0.5*np.tan(u_max[1]))
    turning_radius = L/(np.cos(beta_max) * np.tan(u_max[1]))

    radius = 7.5

    #create a state constraint function

    def h(x):
        """State constraint function"""
        
        radius = 7.5

        return -np.sqrt(x[0]**2 + x[1]**2) + radius

    # create a terminal constraint function
    def cf(x):
        """Terminal constraint function for casadi type arguments"""

        turning_radius = 2*2.8

        return -np.sqrt(x[0]**2 + x[1]**2) + 2*turning_radius

    # set parameters for the CBF module
    T_direct = 2*np.pi*turning_radius/u_min[0]  # time to drive half circle with max speed
    N_direct = 50
    gamma = 0.1

    # set domain bounds
    domain_lower_bound_partial = np.array([-25,0,-np.pi])
    domain_upper_bound_partial = np.array([0,0,0])
    discretization_partial = np.array([61,1,21])
    domain_lower_bound_partial_2 = np.array([-25,0,-np.pi])
    domain_upper_bound_partial_2 = np.array([0,0,np.pi])
    discretization_partial_2 = np.array([61,1,41])
    domain_lower_bound_complete = np.array([-25,-25,-np.pi])
    domain_upper_bound_complete = np.array([25,25,np.pi])
    discretization_complete = np.array([61,31,21])

    print("Number of grid points to be computed: ", np.prod(discretization_partial))

    # create a CBF module
    myCBFmodule_partial = CBFmodule(h=h, 
                            dynamicSystem=myBike, 
                            cf=cf, 
                            T=T_direct, 
                            N=N_direct,
                            gamma=gamma, 
                            domain_lower_bound=domain_lower_bound_partial, 
                            domain_upper_bound=domain_upper_bound_partial, 
                            discretization=discretization_partial,
                            p_norm=50,
                            p_norm_decrement=10,
                            p_norm_min=40)

    ########################################################################################
    # Compute CBF on specified domain
    # explicit construction of the CBF values by analytically solving the optimization problem

    # Simulations horizons for starting points with ||x|| <= 2*turning_radius (innner)
    # and ||x|| > 2*turning_radius (outer)
    T = 2/3 * T_direct
    dt = T/N_direct

    input_1 = np.array([u_min[0]*np.ones(myCBFmodule_partial.N), 
                        u_min[1]*np.ones(myCBFmodule_partial.N)])
    input_2 = np.array([u_min[0]*np.ones(myCBFmodule_partial.N),
                        u_max[1]*np.ones(myCBFmodule_partial.N)])

    inputs= np.array([input_1, input_2])

    # 2. Compute the CBF on the domain

    tic = time.time()

    point_list = myCBFmodule_partial.cbf.getPointList()

    for point in point_list:
        x0 = np.array(point["point"], dtype=float)
        index = point["index"]

        if x0[2] == 0 and x0[0]>-8.0:
            pass

        # If the point is outside of the terminal constraint domain, and points towards the center of the circle, then the CBF value equals h(x0)
        sigma = np.arccos(x0[0]/np.linalg.norm(x0[0:2]))  # angle of the point in the x-y plane
        if x0[1] < 0:
            sigma = -sigma
        Delta = x0[2] - sigma  # Orintation points towarsds the center of the circle if Delta is in [pi/2, 3*pi/2] or [-pi/2, -3*pi/2]

        if np.linalg.norm(x0[0:2]) >= 2*turning_radius and ((Delta >= np.pi/2 and Delta <= 3*np.pi/2) or (Delta <= -np.pi/2 and Delta >= -3*np.pi/2 )):
            ## If the point is outside of the terminal constraint domain and points towards the center of the circle, we can directly compute the CBF value
            myCBFmodule_partial.cbf.cbf_values[index] = h(x0)
        elif (np.linalg.norm(x0[0:2]) <= 2*turning_radius and ((Delta >= np.pi/2 and Delta <= 3*np.pi/2) or (Delta <= -np.pi/2 and Delta >= -3*np.pi/2 ))):
            # case: assign value in terminal set
            myCBFmodule_partial.cbf.cbf_values[index] = h(np.array([-2*turning_radius, 0, 0]))  # compute the CBF value at the point on the circle with radius 2*turning_radius
        else: 
            # Otherwise, we need to compute the CBF value by simulating the system with the defined input trajectories
            h_values = np.zeros(inputs.shape[0])

            ## Simulate system starting in x0 and compute the smallest value of h along the trajectory
            for k in range(inputs.shape[0]):
                # Simulate the system with the given input trajectory
                u_traj = inputs[k]

                time_traj, state_traj = myBike.simulateOverHorizon(x0=x0,
                                                                    u=u_traj,
                                                                    dt=dt)
                
                # Compute the state constraint function h along the trajectory
                h_values_traj = [h(state_traj[:, k]) - gamma*k*dt for k in range(myCBFmodule_partial.N+1)]

                # Store the minimum value of h along the trajectory as candidate for the CBF value
                h_values[k] = np.min(h_values_traj)

            # Store the CBF value in the CBF module
            myCBFmodule_partial.cbf.cbf_values[index] = np.min(np.max(h_values))

            # Upper bound the CBF value with the value within the terminal set
            myCBFmodule_partial.cbf.cbf_values[index] = np.min([myCBFmodule_partial.cbf.cbf_values[index], h(np.array([-2*turning_radius, 0, 0]))])

        print("Computed CBF value for point ", point["point"], " with index ", point["index"], ":", myCBFmodule_partial.cbf.cbf_values[index])

    toc = time.time()
    computation_time_partial_cbf = toc - tic
    print("Computation of the CBF on the partial domain took ", toc-tic, " seconds.")

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

        if norm_x != 0:
            cos_theta = dot_product / (norm_m_vec * norm_x)
            theta_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Clip to avoid NaN due to floating point errors
        else:
            theta_rad = 0.0

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



