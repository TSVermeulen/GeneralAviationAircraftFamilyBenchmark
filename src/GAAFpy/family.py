# -*- coding: utf-8 -*-
"""
family.py
=================

Description
-----------
Implementation of the General Aviation Aircraft (GAA) Family Benchmark Problem 
in Python

This module provides a class-based implementation of the General Aviation Aircraft
(GAA) family design problem, suitable for single- or multi-objective optimisation.

The problem involves designing three aircraft variants (2-, 4-, and 6-seater)
with product platform commonality constraints.

Classes
-------
GAABenchmark
    Class representing the GAA family design problem, handling vectorised
    evaluation of objectives and constraints for multiple solutions.

Examples
--------
A detailed example is included at the bottom of the file demonstrating
how to instantiate the GAABenchmark class, evaluate a single solution, and
perform batch evaluations.

Notes
-----
This module is effectively a translation of the GAA problem as implemented in
the MOEA framework in Java in [5]. The Java implementation, just like this
implementation, is based on the original problem formulation and RSMs from
[1-4].

References
----------
    [1] T. W. Simpson, W. Chen, J. K. Allen, and F. Mistree (1996),
        "Conceptual design of a family of products through the use of
        the robust concept exploration method," in 6th AIAA/USAF/NASA/
        ISSMO Symposium on Multidiciplinary Analysis and Optimization,
        vol. 2, pp. 1535-1545.

    [2] T. W. Simpson, B. S. D'Souza (2004), "Assessing variable levels
        of platform commonality within a product family using a
        multiobjective genetic algorithm," Concurrent Engineering:
        Research and Applications, vol. 12, no. 2, pp. 119-130.

    [3] R. Shah, P. M. Reed, and T. W. Simpson (2011), "Many-objective
        evolutionary optimization and visual analytics for product
        family design," Multiobjective Evolutionary Optimisation for
        Product Design and Manufacturing, Springer, London, pp. 137-159.

    [4] D. Hadka, P. M. Reed, and T. W. Simpson (2012), "Diagnostic
        Assessment of the Borg MOEA on Many-Objective Product Family
        Design Problems," WCCI 2012 World Congress on Computational
        Intelligence, Congress on Evolutionary Computation, Brisbane,
        Australia, pp. 986-995.

    [5] Zatarain Salazar, J., Hadka, D., Reed, P., Seada, H., and Deb, K.
        (2024). Diagnostic benchmarking of many-objective evolutionary
        algorithms for real-world problems. Engineering Optimization,
        1-22. https://doi.org/10.1080/0305215X.2024.2381818.

Versioning
----------
@author: T.S. Vermeulen
@email: T.S.Vermeulen@tudelft.nl
@version: 1.1.5
@date (dd-mm-yyyy): 20-03-2026

Changelog:
- V1.0: Initial version. Tested to function for both single solution and
        batch evaluations. Single solution test shows matching output with MOEA
        Java implementation.
- V1.1: Cleaned up code, split out modules. Prepared for package release.
- V1.1.5: Moved variable bounds to utils. Moved constraint limits to utils and 
          made them optional inputs for more flexible usage. 
"""

# Import standard libraries
from typing import Tuple, Dict, Any

# Import 3rd party libraries
import numpy as np

# Import local modules
from .utils import SCALING_PARAMS, load_rsm_coefficients, CONSTRAINT_LIMITS

# Concrete Classes

class GAABenchmark:
    """
    Main class representing the General Aviation Aircraft family design problem.

    This class handles:
    - Response surface model evaluations for all variants
    - Product family penalty calculation
    - Constraint function evaluations
    - Objective function evaluations
    """

    def __init__(self,
                 design_variables: np.ndarray) -> None:
        """
        Initialise the GAA benchmark problem.

        Args:
            design_variables: np.ndarray
                - Array of shape (N, 27), where N is the number of solutions
        """

        design_variables = np.asarray(design_variables, dtype=float)

        # Convert 1D to 2D if a single solution is requested
        if design_variables.ndim == 1:
            if design_variables.shape[0] != 27:
                raise ValueError(
                    f"1D array must have exactly 27 elements, got {design_variables.shape[0]}"
                )
            design_variables = design_variables.reshape(1, -1)
        elif design_variables.ndim == 2:
            if design_variables.shape[1] != 27:
                raise ValueError(
                    f"2D array must have 27 columns, got {design_variables.shape[1]}"
                )
        else:
            raise ValueError(
                f"design_variables must be 1D or 2D, got {design_variables.ndim}D"
            )

        self.design_variables = design_variables


    def evaluate(self,
                 constraint_targets: Dict[str, Any] | None = None
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate the GAA problem for design solution(s).

        Returns:
            Tuple of (objectives, constraints):
            - objectives: np.ndarray, shape (N, 10) — 10 objectives per solution
            - constraints: np.ndarray, shape (N, 18) — 18 constraint violations per solution
            - summed_CV: np.ndarray, shape (N,) — sum of constraint violations per solution
        """

        # Vectorised evaluation pipeline
        scaled_vars = self._scale_variables(self.design_variables)
        response_vars = self._get_response_variables(scaled_vars)
        objectives = self._calculate_objectives(response_vars, scaled_vars)
        constraints, summed_CV = self._calculate_constraints(
            response_vars,
            constraint_targets if constraint_targets is not None else CONSTRAINT_LIMITS)
        return objectives, constraints, summed_CV


    @staticmethod
    def _scale_variables(design_vectors: np.ndarray) -> np.ndarray:
        """
        Vectorised scaling of design variables for all solutions.

        Args:
            design_vectors: np.ndarray, shape (N, 27)
                Raw design variables

        Returns:
            np.ndarray, shape (N, 27)
                Scaled design variables
        """

        scaled = design_vectors.copy()

        # Apply scaling to each design variable (repeats for all 3 variants)
        for variant_idx in range(3):
            for var_idx, (_, center, scale) in enumerate(SCALING_PARAMS):
                col_idx = variant_idx * 9 + var_idx
                scaled[:, col_idx] = (design_vectors[:, col_idx] - center) / scale

        return scaled


    def _get_response_variables(self,
                                scaled_vars_all: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Vectorised response variable calculation for all solutions and variants.

        Args:
            scaled_vars_all: np.ndarray, shape (N, 27)
                Scaled design variables for all solutions

        Returns:
            Tuple of 3 np.ndarrays (one per variant), each shape (N, 9)
        """

        coeffs_data = load_rsm_coefficients()
        variant_names = ["2-seater", "4-seater", "6-seater"]
        response_names = ["NOISE", "WEMP", "DOC", "ROUGH", "WFUEL", "PURCH", "RANGE", "LDMAX", "VCMAX"]

        responses = [np.zeros((self.design_variables.shape[0], 9)) for _ in range(3)]

        for variant_idx, variant_name in enumerate(variant_names):
            start_col = variant_idx * 9
            end_col = start_col + 9
            variant_vars = scaled_vars_all[:, start_col:end_col]
            coeffs = coeffs_data[variant_name]

            for response_idx, response_name in enumerate(response_names):
                response_coeffs = coeffs[response_name]
                responses[variant_idx][:, response_idx] = (
                    self._rsm_evaluation(response_coeffs, variant_vars)
                )

        return responses[0], responses[1], responses[2]


    @staticmethod
    def _rsm_evaluation(coeffs: Dict[str, Any],
                        variant_vars: np.ndarray) -> np.ndarray:
        """
        Vectorised RSM polynomial evaluation for all solutions.

        Args:
            coeffs: Dict[str, Any]
                Coefficient dictionary with constant, linear, interaction,
                and quadratic terms.
            variant_vars: np.ndarray, shape (N, 9)
                Scaled design variables for N solutions

        Returns:
            np.ndarray, shape (N,)
                Response variable values for all solutions
        """

        var_names = [
            "CSPD", "AR", "SWEEP", "DPROP", "WINGLD",
            "AF", "SEATW", "ELODT", "TAPER",
        ]

        n_solutions = variant_vars.shape[0]
        result = np.full(n_solutions, coeffs["constant"], dtype=float)

        # Linear terms
        for var_idx, var_name in enumerate(var_names):
            if var_name in coeffs["linear"]:
                coeff = coeffs["linear"][var_name]
                result += coeff * variant_vars[:, var_idx]

        # Interaction terms
        # Precompute variable name to index mapping for O(1) lookups
        var_index = {name: i for i, name in enumerate(var_names)}

        for interaction_key, coeff in coeffs["interaction"].items():
            var1_name, var2_name = interaction_key.split(',')
            var1_idx = var_index[var1_name]
            var2_idx = var_index[var2_name]
            result += coeff * variant_vars[:, var1_idx] * variant_vars[:, var2_idx]

        # Quadratic terms
        for var_idx, var_name in enumerate(var_names):
            if var_name in coeffs["quadratic"]:
                coeff = coeffs["quadratic"][var_name]
                result += coeff * variant_vars[:, var_idx] ** 2

        return result


    def _calculate_objectives(self,
                              response_vars_all: Tuple[np.ndarray, np.ndarray, np.ndarray],
                              scaled_vars_all: np.ndarray) -> np.ndarray:
        """
        Vectorised objective calculation for all solutions.

        Args:
            response_vars_all: Tuple of 3 arrays, each (N, 9)
            scaled_vars_all: np.ndarray, shape (N, 27)

        Returns:
            np.ndarray, shape (N, 10)
        """

        objectives = np.zeros((self.design_variables.shape[0], 10))
        response_names = ["NOISE", "WEMP", "DOC", "ROUGH", "WFUEL", "PURCH", "RANGE", "LDMAX", "VCMAX"]

        responses_2s, responses_4s, responses_6s = response_vars_all
        all_responses = [responses_2s, responses_4s, responses_6s]

        # Minimisation objectives - we take the worst case across variants
        obj_indices = {"NOISE": 0, "WEMP": 1, "DOC": 2, "ROUGH": 3, "WFUEL": 4, "PURCH": 5}
        for response_name, obj_idx in obj_indices.items():
            response_idx = response_names.index(response_name)
            for variant_response in all_responses:
                objectives[:, obj_idx] = np.maximum(
                    objectives[:, obj_idx], variant_response[:, response_idx]
                )

        # Maximisation objectives - we take the worst case across variants
        obj_indices_min = {"RANGE": 6, "LDMAX": 7, "VCMAX": 8}
        for response_name, obj_idx in obj_indices_min.items():
            response_idx = response_names.index(response_name)
            objectives[:, obj_idx] = np.full(self.design_variables.shape[0], np.inf)
            for variant_response in all_responses:
                objectives[:, obj_idx] = np.minimum(
                    objectives[:, obj_idx], variant_response[:, response_idx]
                )

        # Platform penalty
        objectives[:, 9] = self._get_platform_penalty(scaled_vars_all)

        return objectives


    @staticmethod
    def _get_platform_penalty(scaled_vars_all: np.ndarray) -> np.ndarray:
        """
        Calculate vectorised platform penalty for all solutions.

        Args:
            scaled_vars_all: np.ndarray, shape (N, 27)

        Returns:
            np.ndarray, shape (N,)
        """

        n_solutions = scaled_vars_all.shape[0]
        penalties = np.zeros(n_solutions)

        for var_idx in range(9):
            var_2s = scaled_vars_all[:, var_idx]
            var_4s = scaled_vars_all[:, 9 + var_idx]
            var_6s = scaled_vars_all[:, 18 + var_idx]

            means = (var_2s + var_4s + var_6s) / 3.0
            penalties += (var_2s - means) ** 2
            penalties += (var_4s - means) ** 2
            penalties += (var_6s - means) ** 2

        return np.sqrt(penalties)


    def _calculate_constraints(self,
                               response_vars_all: Tuple[np.ndarray, np.ndarray, np.ndarray],
                               constraint_targets: Dict[str, Any] = CONSTRAINT_LIMITS) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorised constraint violation calculation for all solutions.

        Args:
            response_vars_all: Tuple of 3 arrays, each (N, 9)
            constraint_targets: Dict[str, Any]
                Dictionary of constraint limits for each response variable.
                If not provided, defaults to CONSTRAINT_LIMITS from utils.py.

        Returns:
            Tuple of np.ndarray, shape (N, 18) and np.ndarray shape (N,)
        """

        constraints = np.zeros((self.design_variables.shape[0], 18))
        response_names = ["NOISE", "WEMP", "DOC", "ROUGH", "WFUEL", "PURCH", "RANGE", "LDMAX", "VCMAX"]
        variant_names = ["2-seater", "4-seater", "6-seater"]
        responses_list = response_vars_all

        constraint_idx = 0
        for variant_idx, variant_name in enumerate(variant_names):
            responses = responses_list[variant_idx]

            # NOISE constraint
            noise_idx = response_names.index("NOISE")
            noise_cv = (responses[:, noise_idx] - constraint_targets["NOISE"]) / constraint_targets["NOISE"]
            constraints[:, constraint_idx] = np.maximum(0, noise_cv)
            constraint_idx += 1

            # WEMP constraint
            wemp_idx = response_names.index("WEMP")
            wemp_cv = (responses[:, wemp_idx] - constraint_targets["WEMP"]) / constraint_targets["WEMP"]
            constraints[:, constraint_idx] = np.maximum(0, wemp_cv)
            constraint_idx += 1

            # DOC constraint
            doc_idx = response_names.index("DOC")
            doc_cv = (responses[:, doc_idx] - constraint_targets["DOC"]) / constraint_targets["DOC"]
            constraints[:, constraint_idx] = np.maximum(0, doc_cv)
            constraint_idx += 1

            # ROUGH constraint
            rough_idx = response_names.index("ROUGH")
            rough_cv = (responses[:, rough_idx] - constraint_targets["ROUGH"]) / constraint_targets["ROUGH"]
            constraints[:, constraint_idx] = np.maximum(0, rough_cv)
            constraint_idx += 1

            # WFUEL constraint
            wfuel_idx = response_names.index("WFUEL")
            wfuel_limit = constraint_targets["WFUEL"][variant_name]
            wfuel_cv = (responses[:, wfuel_idx] - wfuel_limit) / wfuel_limit
            constraints[:, constraint_idx] = np.maximum(0, wfuel_cv)
            constraint_idx += 1

            # RANGE constraint
            range_idx = response_names.index("RANGE")
            range_cv = -(responses[:, range_idx] - constraint_targets["RANGE"]) / constraint_targets["RANGE"]
            constraints[:, constraint_idx] = np.maximum(0, range_cv)
            constraint_idx += 1

        # Compute summed constraint violation to remain in-line with MOEA Java 
        # implementation
        summed_CV = np.sum(constraints, axis=1) 

        return constraints, summed_CV


# Example usage
if __name__ == "__main__":
    import time
    from .utils import VARIABLE_BOUNDS

    print("=" * 70)
    print("GAA benchmark problem - test input and evaluation")
    print("=" * 70)

    # Single solution evaluation
    design_vars = np.array([0.336628, 7.355679, 4.687910, 5.703373, 22.424994,
                            96.004001, 19.145602, 3.212344, 0.460768, 0.240046,
                            7.482713, 4.846902, 5.556927, 23.444377, 92.595728,
                            17.764686, 3.294423, 0.548547, 0.332088, 7.182878,
                            5.689232, 5.510011, 23.308395, 85.131278, 16.495613,
                            3.708131, 0.542831])

    # Instantiate class and evaluate single solution
    gaa = GAABenchmark(design_vars)
    objectives, constraints, summed_CV = gaa.evaluate()

    print("\n--- Single Solution ---")
    objective_names = ["Max Noise",
                       "Max Empty Weight",
                       "Max DOC",
                       "Max Roughness",
                       "Max Fuel Weight",
                       "Max Purchase Cost",
                       "Min Range",
                       "Min L/D Ratio",
                       "Min Cruise Speed",
                       "Platform Penalty"]

    print("Objectives:")
    for i, (name, value) in enumerate(zip(objective_names, objectives[0], strict=True)):
        print(f"  {i + 1}. {name}: {value:.2f}")
    print(f"Max constraint and corresponding ID: {np.max(constraints[0])}, ID: {np.argmax(constraints[0])}")

    # Batch evaluation
    print("\n" + "=" * 70)
    n_solutions = 100
    print("--- Batch Evaluation (2D Array Input of 100 solutions) ---")
    design_vectors = np.random.rand(n_solutions, 27)

    # Scale to valid ranges
    uppers = np.asarray(VARIABLE_BOUNDS[1], dtype=float)
    lowers = np.asarray(VARIABLE_BOUNDS[0], dtype=float)
    design_vectors = lowers + design_vectors * (uppers - lowers)

    # Perform batch analysis
    gaa_batch = GAABenchmark(design_vectors)
    start_time = time.time()
    objectives_batch, constraints_batch, summed_CV_batch = gaa_batch.evaluate()
    vectorised_time = time.time() - start_time

    print(f"Evaluation time: {vectorised_time:.4f} seconds")
    print(f"Objectives shape: {objectives_batch.shape}")
    print(f"Constraints shape: {constraints_batch.shape}")
    print(f"Summed Constraint Violations shape: {summed_CV_batch.shape}")

    print(f"Max objectives across solutions: \n {np.max(objectives_batch, axis=0)} ...")
    print(f"Max constraint violation across solutions: \n {np.max(constraints_batch, axis=0)}")

    # Performance comparison
    print("\n" + "=" * 70)
    print("--- Performance Comparison ---")
    print("Sequential (10 single solutions):")
    start_time = time.time()
    for design_vec in design_vectors[:10]:
        gaa_seq = GAABenchmark(design_vec)
        _, _, _ = gaa_seq.evaluate()
    single_time = time.time() - start_time
    estimated_sequential = single_time * (n_solutions / 10)

    print(f"  10 solutions: {single_time:.4f}s")
    print(f"  Estimated for {n_solutions}: {estimated_sequential:.4f}s")
    print(f"  Vectorised for {n_solutions}: {vectorised_time:.4f}s")
    print(f"  Speedup: {estimated_sequential / vectorised_time if vectorised_time > 0 else 0:.1f}x")
    print("\n" + "=" * 70)
