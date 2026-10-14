#-*- coding: utf-8 -*-
"""
utils.py
========

Description
-----------
Basic utilities for the module. 

Versioning
----------
@author: T.S. Vermeulen
@email: T.S.Vermeulen@tudelft.nl
@version: 1.2
@date (dd-mm-yyyy): 12-10-2026

Changelog:
- V1.0: Initial version.
- V1.1: Added additional variable describing the design variable bounds
- V1.2: Added coefficient caching. Added filtering of coefficients based on 
        requested degree. Added error handling for missing degrees.
"""

# Module Constants
__author__ = "Thomas Stephan Vermeulen"
__copyright__ = "Copyright 2026, all rights reserved"
__status__ = "Release"

# Standard library imports
import os
import json
from typing import Dict, Any, List

# Module-level cache for RSM coefficients
_COEFFICIENT_CACHE: Dict[str, Dict[str, Any]] = {}

_RSM_DEGREE_ORDER = ["linear", "interaction", "quadratic"]

# Design variable scaling parameters (shared across all scaling methods)
# Each tuple: (parameter_name, center, scale)
SCALING_PARAMS = [
    ("CSPD", 0.36, 0.12),
    ("AR", 9, 2),
    ("SWEEP", 3, 3),
    ("DPROP", 5.734, 0.234),
    ("WINGLD", 22, 3),
    ("AF", 97.5, 12.5),
    ("SEATW", 17, 3),
    ("ELODT", 3.375, 0.375),
    ("TAPER", 0.73, 0.27),
]

# Constraint limits
CONSTRAINT_LIMITS = {
    "NOISE": 75,
    "WEMP": 2200,
    "DOC": 80,
    "ROUGH": 2,
    "WFUEL": {"2-seater": 450, "4-seater": 475, "6-seater": 500},
    "RANGE": 2000,
}

CONSTRAINT_SYMBOLS = ["$L_{p_{TO}}$",
                      "$W_{OE}$",
                      "$DOC$",
                      "$K_r$",
                      "$W_{F}$",
                      "$R$"]

# Variant names
VARIANT_NAMES = ["2-seater", "4-seater", "6-seater"]

# Variable names as list
VARIABLE_NAMES = ["CSPD2", "AR2", "SWEEP2", "DPROP2", "WINGLD2", "AF2", "SEATW2", "ELODT2", "TAPER2", 
                  "CSPD4", "AR4", "SWEEP4", "DPROP4", "WINGLD4", "AF4", "SEATW4", "ELODT4", "TAPER4", 
                  "CSPD6", "AR6", "SWEEP6", "DPROP6", "WINGLD6", "AF6", "SEATW6", "ELODT6", "TAPER6"]

# Response variable names as list
RESPONSE_NAMES = ["NOISE", "WEMP", "DOC", "ROUGH", "WFUEL", "PURCH", "RANGE", "LDMAX", "VCMAX"]


# Design variable bounds (raw/unscaled values)
VARIABLE_BOUNDS: List[List[float]] = [[0.24,  # CSPD2
                                       7,     # AR2 
                                       0,     # SWEEP2 
                                       5.5,   # DPROP2
                                       19,    # WINGLD2 
                                       85,    # AF2 
                                       14,    # SEATW2
                                       3,     # ELODT2
                                       0.46,  # TAPER2
                                       0.24,  # CSPD4
                                       7,     # AR4
                                       0,     # SWEEP4
                                       5.5,   # DPROP4 
                                       19,    # WINGLD4
                                       85,    # AF4 
                                       14,    # SEATW4 
                                       3,     # ELODT4 
                                       0.46,  # TAPER4 
                                       0.24,  # CSPD6
                                       7,     # AR6
                                       0,     # SWEEP6
                                       5.5,   # DPROP6
                                       19,    # WINGLD6
                                       85,    # AF6
                                       14,    # SEATW6
                                       3,     # ELODT6 
                                       0.46   # TAPER6
                                       ],  # Lower bounds
                                      [0.48,   # CSPD2
                                       11,     # AR2 
                                       6,      # SWEEP2 
                                       5.968,  # DPROP2
                                       25,     # WINGLD2 
                                       110,    # AF2  
                                       20,     # SEATW2
                                       3.75,   # ELODT2 
                                       1,      # TAPER2
                                       0.48,   # CSPD4 
                                       11,     # AR4
                                       6,      # SWEEP4 
                                       5.968,  # DPROP4  
                                       25,     # WINGLD4 
                                       110,    # AF4      
                                       20,     # SEATW4      
                                       3.75,   # ELODT4      
                                       1,      # TAPER4  
                                       0.48,   # CSPD6     
                                       11,     # AR6 
                                       6,      # SWEEP6 
                                       5.968,  # DPROP6 
                                       25,     # WINGLD6
                                       110,    # AF6 
                                       20,     # SEATW6
                                       3.75,   # ELODT6  
                                       1       # TAPER6 
                                       ],  # Upper bounds
                                     ]


# Module Functions
def load_rsm_coefficients(degree: str = "quadratic") -> Dict[str, Dict[str, Any]]:
    """
    Load RSM coefficients from external JSON file.
    Results are cached module-wide to avoid repeated file I/O during
    optimisation.

    Args:
        degree (str): The highest degree of the RSM to load.
                  Options are "linear", "interaction", or "quadratic".
                  The returned coefficients include the requested degree and
                  all lower-order terms.

    Returns:
        Dictionary of coefficients organised by variant and response variable.

    Raises:
        FileNotFoundError: If coefficients file cannot be found in any location
        json.JSONDecodeError: If coefficients file is invalid JSON
    """

    global _COEFFICIENT_CACHE

    if degree not in _RSM_DEGREE_ORDER:
        raise ValueError(
            f"degree must be one of _RSM_DEGREE_ORDER, got {degree!r}"
        )

    if _COEFFICIENT_CACHE:
        return _filter_rsm_coefficients(_COEFFICIENT_CACHE, degree)

    # List of candidate locations to search
    candidates = []

    # Construct module and folder paths
    module_dir = os.path.dirname(os.path.abspath(__file__))

    # JSON file alongside this module
    candidates.append(os.path.join(module_dir, "rsm_coefficients.json"))

    # relative to current working directory
    candidates.append(os.path.join(os.getcwd(), "rsm_coefficients.json"))
    candidates.append("rsm_coefficients.json")  # Just in case

    # Try each candidate location
    for coeffs_file in candidates:
        if os.path.exists(coeffs_file):
            try:
                with open(coeffs_file, 'r', encoding='utf-8') as f:
                    _COEFFICIENT_CACHE = json.load(f)
                    return _filter_rsm_coefficients(_COEFFICIENT_CACHE, degree)
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(
                    f"Invalid JSON in {coeffs_file}: {e.msg}",
                    e.doc,
                    e.pos
                ) from e

    # If we get here, file was not found in any location
    error_msg = (
        "RSM coefficients file 'rsm_coefficients.json' not found.\n\n"
        "Searched in the following locations:\n"
    )
    for i, path in enumerate(candidates, 1):
        error_msg += f"  {i}. {path}\n"
    error_msg += (
        "\nEnsure the file exists in one of these locations,\n"
        "or run the script from the project root directory."
    )
    raise FileNotFoundError(error_msg)


def _filter_rsm_coefficients(coefficients: Dict[str, Dict[str, Any]],
                             degree: str) -> Dict[str, Dict[str, Any]]:
    """Return a cumulative coefficient view up to the requested degree."""

    if degree == "quadratic":
        return json.loads(json.dumps(coefficients))

    requested_degree_index = _RSM_DEGREE_ORDER.index(degree)
    allowed_degrees = _RSM_DEGREE_ORDER[:requested_degree_index + 1]

    filtered_coefficients: Dict[str, Dict[str, Any]] = {}
    for variant_name, variant_coeffs in coefficients.items():
        filtered_coefficients[variant_name] = {}
        for response_name, response_coeffs in variant_coeffs.items():
            missing_degrees = [
                allowed_degree
                for allowed_degree in allowed_degrees
                if allowed_degree not in response_coeffs
            ]
            if missing_degrees:
                raise ValueError(
                    f"Requested degree '{degree}' is missing terms {missing_degrees} "
                    f"for {variant_name}:{response_name}"
                )

            filtered_response: Dict[str, Any] = {}
            if "constant" in response_coeffs:
                filtered_response["constant"] = response_coeffs["constant"]
            for allowed_degree in allowed_degrees:
                filtered_response[allowed_degree] = response_coeffs[allowed_degree]

            filtered_coefficients[variant_name][response_name] = filtered_response

    return filtered_coefficients


if __name__ == "__main__":
    # Test run the function to verify it works
    output = load_rsm_coefficients()
    print(f"Loaded coefficients for variants: {list(output.keys())}")