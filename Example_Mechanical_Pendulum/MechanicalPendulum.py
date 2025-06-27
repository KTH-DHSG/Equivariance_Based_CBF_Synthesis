"""

    Implementation of the dynamics of the mechanical pendulum example from the CBF paper (in Section IV-A).

    Adrian Wiltz, 2025

"""

import numpy as np
from Dynamics.DynamicSystem import DynamicSystem
import casadi as ca

class MechnicalPendulum(DynamicSystem):
    
    def __init__(self,L,x0=None,u_min=-np.inf,u_max=np.inf, d_m_max=0):
        """Initialization.

        Args:
            A (NumPy array): state matrix 2x2
            B (NumPy array): input matrix 2x2
            x0 (NumPy array with length 2): initial state vector; Default is zero vector with corresponding dimension
            u_min (NumPy array of with length 2): lower bound input constraint; Default is -inf with corresponding dimension
            u_max (NumPy array of with length 2): upper bound input constraint; Default is inf with corresponding dimension
        """

        x_dim = 2
        u_dim = 1

        self.L = L

        if x0 is None:
            x0 = np.zeros(x_dim)

        if not all(value is None for value in [x0,u_min,u_max]):
            # initialization with variables
            super().__init__(x0,x_dim,u_dim,u_min,u_max)
        else:
            # empty initialization of instance, can be used e.g. for loading data from a file
            pass     

    def f(self, x, u):
        """Implementation of the single integrator dynamics using casadi data types.

        Args:
            x (casadi.MX or casadi.SX): current state
            u (casadi.MX or casadi.SX): control input (u0 - orientation of impact, u1 - speed)

        Returns:
            casadi.MX or casadi.SX: time derivative of system state
        """

        x_dot = ca.vertcat(x[1], -9.81/self.L*ca.sin(x[0]) + u)

        return x_dot
    
    def f_disturbed(self, x, u, d_m):
        """Implementation of the single integrator dynamics using casadi data types.

        Args:
            x (casadi.MX or casadi.SX): current state
            u (casadi.MX or casadi.SX): control input (u0 - orientation of impact, u1 - speed)
            d_m (casadi.MX or casadi.SX): disturbance input (d_m - disturbance in x direction)

        Returns:
            casadi.MX or casadi.SX: time derivative of system state
        """

        x_dot_disturbed = ca.vertcat(x[1], -9.81/self.L*ca.sin(x[0]) + u + d_m)

        return x_dot_disturbed
    
    def __str__(self):
        str = "LinearSystem with state matrix A:\n"
        str += str(self.A)  
        str += "\n and input matrix B:\n"
        str += str(self.B)
        str += "\n for norm bounded input constraints\n"
        return str