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
@version: 1.1
@date (dd-mm-yyyy): 19-03-2026

Changelog:
- V1.0: Initial version. Tested to function for both single solution and
        batch evaluations. Single solution test shows matching output with MOEA
        Java implementation.
- V1.1: Improved error handling and documentation. Split out the GAABenchmark 
        class into a separate module (gaa.py) to improve modularity and 
        maintainability.
"""

# Standard library imports
from typing import Dict, Any

# 3rd party imports
import numpy as np

# Module imports
from .utils import SCALING_PARAMS, load_rsm_coefficients

# Module Constants
__author__ = "Thomas Stephan Vermeulen"
__copyright__ = "Copyright 2026, all rights reserved"
__status__ = "Release"


# Concrete Classes

class AircraftVariant:
    """
    Class-based representation of an individual aircraft variant within the GAA
    family design problem. Either represents a 2-seater, 4-seater, or 6-seater
    variant.
    """

    def __init__(self, variant_name: str,
                 design_vars: np.ndarray,
                 variant_index: int) -> None:
        """
        Initialise an aircraft variant.

        Args:
            variant_name: str,
                Name of the variant (e.g., "2-seater", "4-seater", "6-seater")
            design_vars: np.ndarray,
                Array of 9 design variables (raw/unscaled values)
            variant_index: int,
                Index for the variant (0 for 2-seater, 1 for 4-seater, 2 for 6-seater)
        """

        self.name = variant_name
        self.variant_index = variant_index
        self.design_vars_raw = design_vars
        self.scaled_vars = self._scale_design_variables(design_vars)
        self.response_vars: Dict[str, float] = {}


    @staticmethod
    def _scale_design_variables(raw_vars: np.ndarray) -> Dict[str, float]:
        """
        Scale raw design variables to normalised space.

        Args:
            raw_vars: np.ndarray
                Raw design variables

        Returns:
            Dictionary of scaled design variables
        """

        scaled: Dict[str, float] = {}
        for i, (name, center, scale) in enumerate(SCALING_PARAMS):
            scaled[name] = float((raw_vars[i] - center) / scale)

        return scaled


    def calculate_response_variables(self) -> None:
        """
        Calculate all response variables using response surface models.
        Writes all response variables to self.response_vars dictionary.
        """

        v = self.scaled_vars

        # Extract scaled variables for brevity
        cspd = v["CSPD"]
        ar = v["AR"]
        sweep = v["SWEEP"]
        dprop = v["DPROP"]
        wingld = v["WINGLD"]
        af = v["AF"]
        seatw = v["SEATW"]
        elodt = v["ELODT"]
        taper = v["TAPER"]

        # Get variant-specific coefficients
        coeffs = self._get_response_surface_coefficients()

        # Calculate NOISE
        self.response_vars["NOISE"] = self._evaluate_rsm(
            coeffs["NOISE"], cspd, ar, sweep, dprop, wingld, af, seatw, elodt, taper
        )

        # Calculate WEMP (Empty Weight)
        self.response_vars["WEMP"] = self._evaluate_rsm(
            coeffs["WEMP"], cspd, ar, sweep, dprop, wingld, af, seatw, elodt, taper
        )

        # Calculate DOC (Direct Operating Cost)
        self.response_vars["DOC"] = self._evaluate_rsm(
            coeffs["DOC"], cspd, ar, sweep, dprop, wingld, af, seatw, elodt, taper
        )

        # Calculate ROUGH (Ride Roughness)
        self.response_vars["ROUGH"] = self._evaluate_rsm(
            coeffs["ROUGH"], cspd, ar, sweep, dprop, wingld, af, seatw, elodt, taper
        )

        # Calculate WFUEL (Fuel Weight)
        self.response_vars["WFUEL"] = self._evaluate_rsm(
            coeffs["WFUEL"], cspd, ar, sweep, dprop, wingld, af, seatw, elodt, taper
        )

        # Calculate PURCH (Purchase Cost)
        self.response_vars["PURCH"] = self._evaluate_rsm(
            coeffs["PURCH"], cspd, ar, sweep, dprop, wingld, af, seatw, elodt, taper
        )

        # Calculate RANGE
        self.response_vars["RANGE"] = self._evaluate_rsm(
            coeffs["RANGE"], cspd, ar, sweep, dprop, wingld, af, seatw, elodt, taper
        )

        # Calculate LDMAX (Maximum Lift-to-Drag Ratio)
        self.response_vars["LDMAX"] = self._evaluate_rsm(
            coeffs["LDMAX"], cspd, ar, sweep, dprop, wingld, af, seatw, elodt, taper
        )

        # Calculate VCMAX (Maximum Cruise Speed)
        self.response_vars["VCMAX"] = self._evaluate_rsm(
            coeffs["VCMAX"], cspd, ar, sweep, dprop, wingld, af, seatw, elodt, taper
        )


    def _get_response_surface_coefficients(self) -> dict:
        """
        Get response surface model coefficients for this variant from cached data.

        Returns:
            Dictionary with RSM coefficients for each response variable
        """

        variant_names = ["2-seater", "4-seater", "6-seater"]
        if not 0 <= self.variant_index < len(variant_names):
            raise ValueError(
                f"Invalid variant index {self.variant_index}."
                f"Must be 0, 1, or 2."
                )
        variant_key = variant_names[self.variant_index]

        # Load coefficients from external file (cached at module level)
        all_coefficients = load_rsm_coefficients()

        if variant_key not in all_coefficients:
            raise ValueError(
                f"Coefficients not found for variant '{variant_key}'. "
                f"Available variants: {list(all_coefficients.keys())}"
            )

        return all_coefficients[variant_key]


    @staticmethod
    def _evaluate_rsm(coeffs: Dict[str, Any],
                      cspd: float,
                      ar: float,
                      sweep: float,
                      dprop: float,
                      wingld: float,
                      af: float,
                      seatw: float,
                      elodt: float,
                      taper: float) -> float:
        """
        Evaluate response surface model polynomial.

        Args:
            coeffs: Dict[str, Any],
                Coefficient dictionary with linear, interaction,
                and quadratic terms.
            cspd: float,
                Scaled cruise speed design variable.
            ar: float,
                Scaled aspect ratio design variable.
            sweep: float,
                Scaled quarter-chord wing sweep design variable.
            dprop: float,
                Scaled propeller diameter design variable.
            wingld: float,
                Scaled wing loading design variable.
            af: float,
                Scaled propeller activity factor design variable.
            seatw: float,
                Scaled seat width design variable.
            elodt: float,
                Scaled tail cone elongation design variable.
            taper: float,
                Scaled wing taper ratio design variable.

        Returns:
            Response variable value.
        """

        var_dict = {
            "CSPD": cspd,
            "AR": ar,
            "SWEEP": sweep,
            "DPROP": dprop,
            "WINGLD": wingld,
            "AF": af,
            "SEATW": seatw,
            "ELODT": elodt,
            "TAPER": taper,
        }

        result = coeffs["constant"]

        # Linear terms
        for var_name, coeff in coeffs["linear"].items():
            result += coeff * var_dict[var_name]

        # Interaction terms
        # Assumes comma-separated string keys (from JSON)
        for key, coeff in coeffs["interaction"].items():
            var1, var2 = key.split(',')
            result += coeff * var_dict[var1] * var_dict[var2]

        # Quadratic terms
        for var_name, coeff in coeffs["quadratic"].items():
            result += coeff * var_dict[var_name] ** 2

        return result


if __name__ == "__main__":
    # Example usage to evaluate a single variant with test design variables
    from .utils import VARIABLE_BOUNDS

    # Construct 100 sample design vectors for testing
    n_solutions = 100
    design_vectors = np.random.rand(9)  # A variant has 9 dvars

    # Scale to valid ranges
    uppers = np.asarray(VARIABLE_BOUNDS[0][:9], dtype=float)
    lowers = np.asarray(VARIABLE_BOUNDS[1][:9], dtype=float)
    design_vectors = lowers + design_vectors * (uppers - lowers)

    # Instantiate class
    variant = AircraftVariant(variant_name="4-seater", 
                              design_vars=design_vectors, 
                              variant_index=1)
    
    variant.calculate_response_variables()
    print(f"Response variables for {variant.name} variant:")
    print(variant.response_vars)