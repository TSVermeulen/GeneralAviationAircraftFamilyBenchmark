# The General Aviation Aircraft Family Benchmark (GAAFpy)

A Python implementation of the GAA family benchmark problem developed by T.W. Simpson, W. Chen. J.K. Allen, F. Mistree, B.S. D'Souza, R. Shah, P.M. Reed, and D. Hadka.

This repository is fundamentally a Python translation of the RealWorldBenchmarks implementation in the MOEA framework by Zatarain Salazar, J., Hadka, D., Reed, P., Seada, H., & Deb, K (see reference 5).

## Requirements

The framework is tested to work using the following Python version and packages:

1. Python >= 3.10
2. Numpy
3. (optional) Pymoo >= 0.6

## Installation

The recommended method is through PyPi by running the command:

```console
pip install gaafpy
```

To install the optional Pymoo multi-objective optimisation package used in the example, use:

```console
pip install -U pymoo
```

## Example usage

A simple example, integrating the GAAFpy family implementation into a Pymoo NSGA-II optimisation problem, using vectorised evaluation:

```python
# Import 3rd party libraries
import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.optimize import minimize

# Import the relevant GAAFpy classes
from GAAFpy.family import GAABenchmark
from GAAFpy.utils import VARIABLE_BOUNDS


# Define the problem class for the GAA benchmark, inheriting from Pymoo's Problem class
class GAABenchmarkProblem(Problem):
    """
    GAA benchmark Pymoo problem definition with 27 design variables,
    10 objectives and 18 constraints. Evaluated using vectorised evaluation,
    where the _evaluate() method retrieves a set of solutions.
    """

    def __init__(self) -> None:
        """Initialise problem definition"""

        super().__init__(n_var=27,
                         n_obj=10,
                         n_ieq_constr=18,
                         xl=np.array(VARIABLE_BOUNDS[0]),
                         xu=np.array(VARIABLE_BOUNDS[1]),
                         )

    def _evaluate(self, x, out, *args, **kwargs) -> None:
        """Evaluate the problem"""

        # Create benchmark object and evaluate the design vector(s)
        benchmark = GAABenchmark(x)
        objectives, constraints, combined_CV = benchmark.evaluate()

        # Objectives need to be formatted to handle the
        # minimisation & maximisation objectives
        objectives_sign_convention = np.array([
            1,   # Minimise noise
            1,   # Minimise empty weight
            1,   # Minimise DOC
            1,   # Minimise roughness
            1,   # Minimise fuel weight
            1,   # Minimise purchase cost
            -1,  # Maximise range
            -1,  # maximise L/D
            -1,  # Maximise Vcruise_max
            1    # Minimise PFPF
        ])
        objectives = objectives * objectives_sign_convention

        # Write the objectives and constraints to the output dictionary
        out["F"] = objectives
        out["G"] = constraints


# Construct the problem and algorithm, using mostly default operators for NSGA-II
problem = GAABenchmarkProblem()
algorithm = NSGA2(pop_size=1000, eliminate_duplicates=True)

# Define termination after 100 generations
termination = get_termination('n_gen', 100)

# Run the optimisation and print the best solutions found
res = minimize(
    problem,
    algorithm,
    termination,
    seed=42,
    save_history=True,
    verbose=True
)

print("Best solution found:")
print(f"X = {res.X}")
print(f"F = {res.F}")
print(f"G = {res.G}")
```

## Validation

Validation of the implemented GAAFpy package was done using the outputs from the provided Example Java file in the MOEA RealWorldBenchmarks repository.
This example outputs 100 solutions to the GAA problem obtained using the NSGAII algorithm after 10,000 generations.
The resulting validation report shows a maximum relative difference of 3.33e-06, a mean relative difference of 1.79e-07, and a std dev of differences of 3.69e-07.
For each of the objectives, the validation performance is summarised in the below table:

| Objective    | Max Relative Difference   | Mean Relative Difference  |
|--------------|---------------------------|---------------------------|
| max_NOISE    | 3.21e-08                  | 1.24e-08                  |
| max_WEMP     | 2.79e-08                  | 7.90e-09                  |
| max_DOC      | 2.09e-06                  | 9.32e-07                  |
| max_ROUGH    | 2.68e-07                  | 1.35e-07                  |
| max_WFUEL    | 1.33e-07                  | 3.86e-08                  |
| max_PURCH    | 4.08e-08                  | 1.20e-08                  |
| min_RANGE    | 3.90e-07                  | 1.25e-07                  |
| min_LDMAX    | 1.91e-07                  | 9.44e-08                  |
| min_VCMAX    | 3.05e-08                  | 1.10e-08                  |
| PFPF         | 3.33e-06                  | 4.24e-07                  |

The 100 solutions obtained from the MOEA framework show zero constraint violations across the 18 constraints present in the problem.
This behaviour is replicated by this Python implementation.

## Community Guidelines

This software is currently being maintained by me @TSVermeulen. If you find any bugs, want to contribute or have any questions, you can [open an issue on GitHub](https://github.com/TSVermeulen/GeneralAviationAircraftFamilyBenchmark/issues).

## License

The Benchmark problem is copyright by the respective authors. Please cite them as appropriate if using the benchmark problem.

## References

1. T. W. Simpson, W. Chen, J. K. Allen, and F. Mistree (1996). "Conceptual design of a family
   of products through the use of the robust concept exploration method." In 6th AIAA/USAF/NASA/
   ISSMO Symposium on Multidiciplinary Analysis and Optimization, vol. 2, pp. 1535-1545.
   ([Link](http://www.researchgate.net/publication/236735937_Conceptual_Design_of_a_Family_of_Products_Through_the_Use_of_the_Robust_Concept_Exploration_Method))

2. T. W. Simpson, B. S. D'Souza (2004). "Assessing variable levels of platform commonality within
   a product family using a multiobjective genetic algorithm." Concurrent Engineering:
   Research and Applications, vol. 12, no. 2, pp. 119-130.
   ([Link](http://cer.sagepub.com/content/12/2/119.abstract))

3. R. Shah, P. M. Reed, and T. W. Simpson (2011). "Many-objective evolutionary optimization and
   visual analytics for product family design." Multiobjective Evolutionary Optimisation for
   Product Design and Manufacturing, Springer, London, pp. 137-159.
   ([Link](http://link.springer.com/chapter/10.1007/978-0-85729-652-8_4))

4. D. Hadka, P. M. Reed, and T. W. Simpson (2012). "Diagnostic Assessment of the Borg MOEA on
   Many-Objective Product Family Design Problems."  WCCI 2012 World Congress on Computational
   Intelligence, Congress on Evolutionary Computation, Brisbane, Australia, pp. 986-995.
   ([Link](http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=6256466))

5. Zatarain Salazar, J., Hadka, D., Reed, P., Seada, H., & Deb, K. (2024). Diagnostic benchmarking
   of many-objective evolutionary algorithms for real-world problems. Engineering Optimization, 1–22. ([Link](https://doi.org/10.1080/0305215X.2024.2381818))
