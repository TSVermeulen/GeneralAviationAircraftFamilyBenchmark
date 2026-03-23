# -*- coding: utf-8 -*-
"""
example_variant.py
=================

Description
-----------
Example application of the GAAFpy package to solve the 
General Aviation Aircraft (GAA) variant design problem using NSGA-II.

This problem is a subpart of the GAA family design problem, where a single 
variant is optimised in isolation. Either a 2-seater, 4-seater, or 6-seater 
variant can be represented.

Classes
-------
GAAVariantBenchmarkProblem
    Class representing the GAA variant design problem, inheriting from Pymoo's 
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
from GAAFpy.variant import AircraftVariant
from GAAFpy.utils import VARIABLE_BOUNDS

# Define module constant which determines which variant is optimised.
# 0 = 2-seater, 1 = 4-seater, 2 = 6-seater
VARIANT_TYPE = 0


# Define the problem class for the GAA benchmark, inheriting from Pymoo's 
# Problem class
class GAAVariantBenchmarkProblem(Problem):
    """ 
    GAA variant benchmark Pymoo problem definition with 9 design variables, 
    9 objectives and 6 constraints. Evaluated using vectorised evaluation,
    where _evaluate retrieves a set of solutions
    """

    def __init__(self) -> None:
        """ Initialise problem definition """
        start_idx = VARIANT_TYPE * 9
        end_idx = start_idx + 9

        super().__init__(n_var=9, 
                         n_obj=9,
                         n_ieq_constr=6,
                         xl=np.array(VARIABLE_BOUNDS[0][start_idx:end_idx]),
                         xu=np.array(VARIABLE_BOUNDS[1][start_idx:end_idx])
                         )
        
    def _evaluate(self, x, out, *args, **kwargs) -> None:
        """ Evaluate the problem """

        # Create variant benchmark object and evaluate the design vector(s)
        benchmark = AircraftVariant(design_vars=x,
                                    variant_index=VARIANT_TYPE)
        
        objectives, constraints, _combined_CV = benchmark.evaluate()

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
                                               -1]) # Maximise Vcruise_max
        
        objectives = objectives * objectives_sign_convention

        # Write the objectives and constraints to the output dictionary 
        out["F"] = objectives
        out["G"] = constraints

if __name__ == "__main__":
    # Construct the problem and algorithm, using mostly default operators for NSGA-II
    problem = GAAVariantBenchmarkProblem()
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
 