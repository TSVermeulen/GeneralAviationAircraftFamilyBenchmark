# -*- coding: utf-8 -*-
"""
example_family.py
=================

Description
-----------
Example application of the GAAFpy package to solve the 
General Aviation Aircraft (GAA) family design problem using NSGA-II.

Implementation of the General Aviation Aircraft (GAA) Family Benchmark Problem 
in Python. The problem involves designing three aircraft variants 
(2-, 4-, and 6-seater) with product platform commonality constraints.

Classes
-------
GAABenchmarkProblem
    Class representing the GAA family design problem, inheriting from Pymoo's 
    Problem class.

Versioning
----------
@author: T.S. Vermeulen
@email: T.S.Vermeulen@tudelft.nl
@version: 1.0
@date (dd-mm-yyyy): 23-03-2026

Changelog:
- V1.0: Initial version.
"""

# Module Constants
__author__ = "Thomas Stephan Vermeulen"
__copyright__ = "Copyright 2026, all rights reserved"
__status__ = "Release"

# Import 3rd party libraries
import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.optimize import minimize

# Import the relevant GAAFpy classes
from GAAFpy.family import GAABenchmark
from GAAFpy.utils import VARIABLE_BOUNDS


# Define the problem class for the GAA benchmark, inheriting from Pymoo's 
# Problem class
class GAABenchmarkProblem(Problem):
    """ 
    GAA benchmark Pymoo problem definition with 27 design variables, 
    10 objectives and 18 constraints. Evaluated using vectorised evaluation,
    where _evaluate retrieves a set of solutions
    """

    def __init__(self) -> None:
        """ Initialise problem definition """

        super().__init__(n_var=27, 
                         n_obj=10,
                         n_ieq_constr=18,
                         xl=np.array(VARIABLE_BOUNDS[0]),
                         xu=np.array(VARIABLE_BOUNDS[1])
                         )
        
    def _evaluate(self, x, out, *args, **kwargs) -> None:
        """ Evaluate the problem """

        # Create benchmark object and evaluate the design vector(s)
        benchmark = GAABenchmark(x)
        objectives, constraints, combined_CV = benchmark.evaluate()

        # Objectives need to be formatted to handle the 
        # minimisation & maximisation objectives
        objectives_sign_convention = np.array([1,   # Minimise noise
                                               1,   # Minimise empty weight
                                               1,   # Minimise DOC
                                               1,   # Minimise roughness
                                               1,   # Minimise fuel weight
                                               1,   # Minimise purchase cost
                                               -1,  # Maximise range
                                               -1,  # maximise L/D
                                               -1,  # Maximise Vcruise_max
                                               1])  # Minimise PFPF
        objectives = objectives * objectives_sign_convention

        # Write the objectives and constraints to the output dictionary 
        out["F"] = objectives
        out["G"] = constraints


# Construct the problem and algorithm, using mostly default operators for NSGA-II
problem = GAABenchmarkProblem()
algorithm = NSGA2(pop_size=1000,
                  eliminate_duplicates=True)  
    
# Define termination after 100 generations
termination = get_termination('n_gen', 100)

# Run the optimisation and print the best solutions found
res = minimize(problem,
               algorithm,
               termination,
               seed=42,
               save_history=True,
               verbose=True)
    
print("Best solution found: \nX = %s\nF = %s\nG = %s" % (res.X, res.F, res.G))
 