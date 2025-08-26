README
======

What the code is about
======================

The code provides exemplary implementations for the CBF synthesis leveraging equivariances and symmetries. Based on partially known CBFs, which are computed using our earlier Predictive_CBF_synthesis_Toolbox from the article

Adrian Wiltz, Dimos V. Dimarogonas. "Predictive Synthesis of Control Barrier Functions and its Application to Time-Varying Constraints", 2025.

the here provided code computes CBFs for a broad variety of different constraints. The theoretical basis for this code is developed in the article

Adrian Wiltz, Dimos V. Dimarogonas. "Leveraging Equivariances and Symmetries in the Control Barrier FUnction Synthesis", 2025.


How to cite this code
=====================

If you find this code usefule, please reference the following paper.

Adrian Wiltz, Dimos V. Dimarogonas. "Leveraging Equivariances and Symmetries in the Control Barrier Function Synthesis", 2025.


Prerequisits
============
-- Casadi, at least version 3.6.0 recommended (optimal control toolbox)
-- dask, recommended version 2023.11.0 (parallel computation toolbox)

-- Predictive_CBF_synthesis_Toolbox; already included in a folder

The code has been tested with Python 3.11.7 (conda).


Structure of the code
=====================

The repository is structured into mainly three parts:

-- Equivariance_Module: Functions for computing CBFs based on equivariances and an at least partially known CBF

-- Multiple folders with examples:
    * Example_Linear_System_I
    * Example_Linear_System_II
    * Example_Mechanical_Pendulum
    * Examples_Predicitve_CBF_Toolbox_revisited
    * Examples_without_symmetry

-- Predictive_CBF_synthesis_Toolbox: Toolbox for the predicition based computation of the partially known CBF

and some additional auxiliary functions in the folder math_aux.


How to get started
==================

Any of the examples can be directly run. The CBFs computed in the paper are readily provided and can be visualized using the visualization scripts. 


List of content
===============

-- Equivariance_Module

    * equi_cbf.py: Functions for the equivariance based CBF synthesis of CBFs. The methods expect as argument the transfromation under which the dynamics are equivariant and which shifts the (partially) known CBF along the boundary of the constraint set. A parallelized version of the function is available. It is beneficial for very large domains, yet for smaller domains, it may introduce a large process management overhead. 

    * multi_equi_cbf.py: Same functionality as equi_cbf.py, however, allows to construct CBFs when multiple transfromations are needed for the construction of the CBF.

-- Example_Linear_System_I: 
    >> Implementation of Example 2 from the paper in Section IV-C

    * Data: Folder containing the results for the CBF considered in Example 2 of the paper

    * cbf_L1.py: Direct computation of CBF for Example 2 in the paper to confirm the theoretically found symmetries.

    * visualization_cbf_L1.py: script for the visualization of the computed CBF

-- Example_Linear_System_II: 
    >> Implementation of the rotationally equivariant system considered in Section IV-C of the paper

    * Data: Folder containing the results for the CBF considered in the correspodnign example of the paper

    * cbf_L2_equi_parallelized.py: equivariance based CBF synthesis using parallelization also for the application of the equivariances 

    * cbf_L2_equi.py: equivariance based CBF synthesis, parallelization only used in the computation of the exlcitly computed points

    * cbf_L2.py: direct computation of the CBF (parallized, based on the Predictive_CBF_synthesis_Toolbox)

    * LinearSystem_L2.py: implementation of the linear system

    * visualization_cbf_L2.py: script for the visualization of the computed CBFs

-- Example_Mechanical_Pendulum:
    >> Implementation of the example on the mechnaical pendulum in Section IV-A of the paper

    * Data: Folder containing the results for the CBF considered in the correspodnign example of the paper

    * cbf_Mechanical Pendulum.py: direct computation of the CBF confirming the theoretically found symmetries

    * MechanicalPendulum.py: Implemetnation of the dynamics of the mechanical pendulum

    * visualization_cbf_mechanical_pendulum.py: script for cbf visualization

-- Examples_Predicitve_CBF_Toolbox_revisited:
    >> Recomputation of the CBFs from the paper 
    Adrian Wiltz, Dimos V. Dimarogonas. "Predictive Synthesis of Control Barrier Functions and its Application to Time-Varying Constraints", 2025.
    by using equivariances for a comparision of the performance. All systems under consideration are input constraint. The results of this simulation study are presented in Section VI-A of the paper.

    ** B1
        >> Kinematic bicycle model with circular obstacle
        
        * Data: folder containing data of CBFs presented in the paper
        * cbf_b1_1_equi.py: equivariacne based cbf synthesis (less agile system)
        * cbf_b1_1.py: direct cbf computation (less agile system)
        * cbf_b1_2_equi.py: equivariacne based cbf synthesis (more agile system)
        * cbf_b1_2.py: direct cbf computation (more agile system)
        * compare_cbfs.py: comparision of CBFs computed via equivariances and the directly computed ones
        * visualize_cbf.py: script for visualizing the computed CBFs

    ** D1
        >> Double integrator with circular obstacle

        * Data: folder containing data of CBFs presented in the paper
        * cbf_d2_equi.py: equivariance based cbf synthesis
        * cbf_d2.py: direct cbf computation

    ** S1
        >> Single integrator with circular constraints (s1 corresponds to the first considered single integrator, s2 is the second single integrator with v_x > 0)

        * Data: folder containing data of CBFs presented in the paper
        * cbf_s1_equi.py: equivariance based cbf synthesis 
        * cbf_s1.py: direct cbf computation 
        * cbf_s2_equi.py: equivariance based cbf synthesis 
        * cbf_s2.py: direct cbf computation 
        * visualize_cbf.py: script for visualizing the computed CBFs

    ** U1
        >> Unicycle dynamics

        * Data: folder containing data of CBFs presented in the paper
        * cbf_u1_equi.py: equivariance based CBF synthsesis
        * cbf_u1.py: direct cbf synthesis
        * visualize_cbf: script for the visualization of the computed CBF

-- Examples_without_symmetry:
    >> Equivariance based CBF computation for the examples presented in Section VI-B of the paper. The dynamics under consideration are the kinematic bicyle model (less agile). The constraints under consideration do not necessarily exhibit the symmetries induced by those transformations under which the dynamics are equivaraint. The CBf synthesis is conducted based on a partially known CBF.

    * Data: Folder containing the data of the completely computed CBFs as considered in the paper
    * Data_cbf_partial: data of the partially computed CBFs
    * Data_large_files: not included as the size of the files exceeds the limit of GitHub; please contact the authors

    >> The following files are for the equivariance based cbf synthesis
    * cbf_circle.py: convex circular obstacle
    * cbf_convex_obstacle.py: advanced convex obstacle
    * cbf_ellipse.py: convex elliptical obstacle
    * cbf_inner_circle_complete.py: circular feasible set
    * cbf_nonconvex_obstacle.py: advanced nonconvex obstacle
    * cbf_square.py: square as obstacle (obstacle with corners)
    * cbf_straight_line_complete.py: half plane constraint

    >> The following files are for the computation of a partially known CBF (based on an explicit construction)
    * cbf_inner_circle_partial.py: partial computation of the cbf for feasible set 
    * cbf_straight_line_partial.py: partial computation of the cbf for the half plane constraint

    >> The following files contain scripts for the visualization of the previously computed CBFs 
    * visualize_cbf_circle.py
    * visualize_cbf_convex_obstacle.py
    * visualize_cbf_ellipse.py
    * visualize_cbf_inner_cirlce_complete.py
    * visualize_cbf_inner_cirlce_partial.py
    * visualize_cbf_nonconvex_obstacle.py
    * visualize_cbf_square.py
    * visualize_cbf_straight_line_complete.py
    * visualize_cbf_straight_line_partial.py

-- math_aux:
    >> auxiliary functions

    * math_aux.py

-- Predictive_CBF_synthesis_Toolbox
    >> Toolbox for the predictive synthesis of CBFs. Please refer to its GitHub page and README for furhter details.


Further documentation
=====================

Further documentation is provided via comments directly in the code. An elaborate documentation is given in the beginning of each class and function. 


Related paper
=============

Adrian Wiltz, Dimos V. Dimarogonas. "Leveraging Equivariances and Symmetries in the Control Barrier FUnction Synthesis", 2025.
