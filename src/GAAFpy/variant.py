# -*- coding: utf-8 -*-
"""
variant.py
=================

Description
-----------
General Aviation Aircraft (GAA) implementation of an individual aircraft
in Python

This module provides a class-based implementation of the General Aviation Aircraft
(GAA) family aircraft variant, suitable for single- or multi-objective optimisation.
Either a a 2-seater, 4-seater, or 6-seater variant can be represented.

Classes
-------
AircraftVariant
    Represents an individual aircraft variant, handling design variable scaling
    and response variable calculations using response surface models (RSMs).

Notes
-----
This module is effectively a translation of the GAA problem as implemented in
the MOEA framework in Java in [5]. The Java implementation, just like this
implementation, is based on the original problem formulation and RSMs from
[1-4].

Note that this module currently only works for single solution evaluations. 

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
@date (dd-mm-yyyy): 23-03-2026

Changelog:
- V1.0: Initial version. Tested to function for both single solution and
        batch evaluations. Single solution test shows matching output with MOEA
        Java implementation.
- V1.1: Improved error handling and documentation. Split out the GAABenchmark 
        class into a separate module (gaa.py) to improve modularity and 
        maintainability.
- V1.1.5: Enabled vectorised evaluation. Cleaned up module. 
"""

# Module Constants
__author__ = "Thomas Stephan Vermeulen"
__copyright__ = "Copyright 2026, all rights reserved"
__status__ = "Release"

# Standard library imports
from typing import Dict, Any, ClassVar, List, Tuple

# 3rd party imports
import numpy as np

# Module imports
from .utils import SCALING_PARAMS, load_rsm_coefficients, CONSTRAINT_LIMITS 

# Concrete Classes

class AircraftVariant:
    """
    Class-based representation of an individual aircraft variant within the GAA
    family design problem. Either represents a 2-seater, 4-seater, or 6-seater
    variant.
    """

    RESPONSE_NAMES: ClassVar[List[str]] = [
        "NOISE", "WEMP", "DOC", "ROUGH", "WFUEL", "PURCH", "RANGE", "LDMAX", "VCMAX",
    ]
    VARIANT_NAMES: ClassVar[List[str]] = ["2-seater", "4-seater", "6-seater"]


    def __init__(self,
                 design_vars: np.ndarray,
                 variant_index: int) -> None:
        """
        Initialise an aircraft variant.

        Args:
            design_vars: np.ndarray,
                Array of 9 design variables (raw/unscaled values)
            variant_index: int,
                Index for the variant (0 for 2-seater, 1 for 4-seater, 2 for 6-seater)
        """

        # Validate variant index and set the variant name
        if not 0 <= variant_index < len(self.VARIANT_NAMES):
            raise ValueError(f"Invalid variant index {variant_index}. "
                             f"Must be 0, 1, or 2.")
        self.name = self.VARIANT_NAMES[variant_index]
    
        # Validate design variables and reshape if needed
        design_vars = np.asarray(design_vars, dtype=float)
        if design_vars.ndim == 1:
            if design_vars.shape[0] != 9:
                raise ValueError(f"1D array must have exactly 9 elements, "
                                 f"got {design_vars.shape[0]}")
            design_vars = design_vars.reshape(1, -1)
        elif design_vars.ndim == 2:
            if design_vars.shape[1] != 9:
                raise ValueError(f"2D array must have 9 columns, "
                                 f"got {design_vars.shape[1]}")
        else:
            raise ValueError(f"design_vars must be 1D or 2D, "
                             f"got {design_vars.ndim}D")

        # Scale the design variables to the normalised space used by the RMSs
        self.scaled_vars = self._scale_design_variables(design_vars)


    def evaluate(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate the aircraft variant.

        Returns:
            Tuple of (objectives, constraints, summed_CV):
            - objectives: np.ndarray, shape (N, 10)
            - constraints: np.ndarray, shape (N, 6)
            - summed_CV: np.ndarray, shape (N,)
        """

        # Calculate the RSM response variables for the design solution(s)
        # These are equal to the objectives. 
        objectives = self._get_response_variables(self.scaled_vars)

        # Use the response variables to compute the constraints
        constraints, summed_cv = self._calculate_constraints(objectives)

        return objectives, constraints, summed_cv


    @staticmethod
    def _scale_design_variables(raw_vars: np.ndarray) -> np.ndarray:
        """
        Scale the raw design variables to the normalised space.

        Args:
            raw_vars: np.ndarray
                Array of the raw design variables

        Returns:
            Scaled design variables of shape (N, 9)
        """

        scaled = raw_vars.copy()
        for i, (_, center, scale) in enumerate(SCALING_PARAMS):
            scaled[:, i] = (raw_vars[:, i] - center) / scale

        return scaled


    def calculate_response_variables(self) -> None:
        """
        Calculate all response variables using response surface models.
        Writes all response variables to self.response_vars dictionary.
        """

        responses = self._get_response_variables(self.scaled_vars)
        self.response_vars = {}

        for response_idx, response_name in enumerate(self.RESPONSE_NAMES):
            values = responses[:, response_idx]
            self.response_vars[response_name] = (
                float(values[0]) if values.shape[0] == 1 else values
            )


    def _get_response_variables(self, scaled_vars: np.ndarray) -> np.ndarray:
        """
        Calculate all response variables for all provided solutions.

        Args:
            scaled_vars: np.ndarray, shape (N, 9)

        Returns:
            np.ndarray, shape (N, 9)
        """

        coeffs = self._get_response_surface_coefficients()
        responses = np.zeros((scaled_vars.shape[0], len(self.RESPONSE_NAMES)))

        for response_idx, response_name in enumerate(self.RESPONSE_NAMES):
            responses[:, response_idx] = self._evaluate_rsm(coeffs[response_name], scaled_vars)

        return responses


    def _calculate_constraints(self, response_vars: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate constraint violations for this variant.

        Args:
            response_vars: np.ndarray, shape (N, 9)

        Returns:
            Tuple of np.ndarray, shape (N, 6) and np.ndarray shape (N,)
        """

        constraints = np.zeros((response_vars.shape[0], 6))

        noise_idx = self.RESPONSE_NAMES.index("NOISE")
        noise_cv = (response_vars[:, noise_idx] - CONSTRAINT_LIMITS["NOISE"]) / CONSTRAINT_LIMITS["NOISE"]
        constraints[:, 0] = np.maximum(0, noise_cv)

        wemp_idx = self.RESPONSE_NAMES.index("WEMP")
        wemp_cv = (response_vars[:, wemp_idx] - CONSTRAINT_LIMITS["WEMP"]) / CONSTRAINT_LIMITS["WEMP"]
        constraints[:, 1] = np.maximum(0, wemp_cv)

        doc_idx = self.RESPONSE_NAMES.index("DOC")
        doc_cv = (response_vars[:, doc_idx] - CONSTRAINT_LIMITS["DOC"]) / CONSTRAINT_LIMITS["DOC"]
        constraints[:, 2] = np.maximum(0, doc_cv)

        rough_idx = self.RESPONSE_NAMES.index("ROUGH")
        rough_cv = (response_vars[:, rough_idx] - CONSTRAINT_LIMITS["ROUGH"]) / CONSTRAINT_LIMITS["ROUGH"]
        constraints[:, 3] = np.maximum(0, rough_cv)

        wfuel_idx = self.RESPONSE_NAMES.index("WFUEL")
        wfuel_limit = CONSTRAINT_LIMITS["WFUEL"][self.name]
        wfuel_cv = (response_vars[:, wfuel_idx] - wfuel_limit) / wfuel_limit
        constraints[:, 4] = np.maximum(0, wfuel_cv)

        range_idx = self.RESPONSE_NAMES.index("RANGE")
        range_cv = -(response_vars[:, range_idx] - CONSTRAINT_LIMITS["RANGE"]) / CONSTRAINT_LIMITS["RANGE"]
        constraints[:, 5] = np.maximum(0, range_cv)

        summed_cv = np.sum(constraints, axis=1)

        return constraints, summed_cv


    def _get_response_surface_coefficients(self) -> dict:
        """
        Get response surface model coefficients for this variant from cached data.

        Returns:
            Dictionary with RSM coefficients for each response variable
        """

        # Load coefficients from external file (cached at module level)
        all_coefficients = load_rsm_coefficients()

        if self.name not in all_coefficients:
            raise ValueError(
                f"Coefficients not found for variant '{self.name}'. "
                f"Available variants: {list(all_coefficients.keys())}"
            )

        return all_coefficients[self.name]


    @staticmethod
    def _evaluate_rsm(coeffs: Dict[str, Any],
                      variant_vars: np.ndarray) -> np.ndarray:
        """
        Evaluate response surface model polynomial.

        Args:
            coeffs: Dict[str, Any],
                Coefficient dictionary with linear, interaction,
                and quadratic terms.
            variant_vars: np.ndarray, shape (N, 9)
                Scaled design variables for N solutions.

        Returns:
            Response variable values for N solutions.
        """

        var_names = [
            "CSPD", "AR", "SWEEP", "DPROP", "WINGLD",
            "AF", "SEATW", "ELODT", "TAPER",
        ]
        var_index = {name: idx for idx, name in enumerate(var_names)}

        n_solutions = variant_vars.shape[0]
        result = np.full(n_solutions, coeffs["constant"], dtype=float)

        # Linear terms
        for var_name, coeff in coeffs["linear"].items():
            var_idx = var_index[var_name]
            result += coeff * variant_vars[:, var_idx]

        # Interaction terms
        # Assumes comma-separated string keys (from JSON)
        for key, coeff in coeffs["interaction"].items():
            var1, var2 = key.split(',')
            var1_idx = var_index[var1]
            var2_idx = var_index[var2]
            result += coeff * variant_vars[:, var1_idx] * variant_vars[:, var2_idx]

        # Quadratic terms
        for var_name, coeff in coeffs["quadratic"].items():
            var_idx = var_index[var_name]
            result += coeff * variant_vars[:, var_idx] ** 2

        return result


if __name__ == "__main__":
    # Example usage to evaluate a single variant with test design variables
    from GAAFpy.utils import VARIABLE_BOUNDS

    # Construct sample design vector for testing
    variant = "4-seater"
    variant_idx = 1
    n_solutions = 1
    design_vectors = np.random.rand(n_solutions, 9)[0]  # A variant has 9 dvars

    # Scale to valid ranges
    start = variant_idx * 9
    end = start + 9
    uppers = np.asarray(VARIABLE_BOUNDS[1][start:end], dtype=float)
    lowers = np.asarray(VARIABLE_BOUNDS[0][start:end], dtype=float)
    design_vectors = lowers + design_vectors * (uppers - lowers)

    # Instantiate class
    variant = AircraftVariant(variant_name=variant, 
                              design_vars=design_vectors, 
                              variant_index=variant_idx)
    
    variant.calculate_response_variables()
    print(f"Response variables for {variant.name} variant:")
    print(variant.response_vars)