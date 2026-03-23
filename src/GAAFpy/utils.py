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
@version: 1.1
@date (dd-mm-yyyy): 20-03-2026

Changelog:
- V1.0: Initial version.
- V1.1: Added additional variable describing the design variable bounds
"""

# Module Constants
__author__ = "Thomas Stephan Vermeulen"
__copyright__ = "Copyright 2026, all rights reserved"
__status__ = "Release"

# Standard library imports
import os
import json
from typing import Dict, Any, List

# Module-level cache for RSM coefficients (loaded once, reused for all optimisations)
_COEFFICIENT_CACHE: Dict[str, Dict[str, Any]] = {}

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
def load_rsm_coefficients() -> Dict[str, Dict[str, Any]]:
    """
    Load RSM coefficients from external JSON file.
    Results are cached module-wide to avoid repeated file I/O during
    optimisation.

    Returns:
        Dictionary of coefficients organised by variant and response variable.

    Raises:
        FileNotFoundError: If coefficients file cannot be found in any location
        json.JSONDecodeError: If coefficients file is invalid JSON
    """

    global _COEFFICIENT_CACHE

    if _COEFFICIENT_CACHE:
        return _COEFFICIENT_CACHE

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
                    return _COEFFICIENT_CACHE
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


if __name__ == "__main__":
    # Test run the function to verify it works
    output = load_rsm_coefficients()
    print(f"Loaded coefficients for variants: {list(output.keys())}")