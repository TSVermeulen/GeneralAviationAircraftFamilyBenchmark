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
@version: 1.0
@date (dd-mm-yyyy): 19-03-2026

Changelog:
- V1.0: Initial version.
"""

# Module Constants
__author__ = "Thomas Stephan Vermeulen"
__copyright__ = "Copyright 2026, all rights reserved"
__status__ = "Release"

# Standard library imports
import os
import json
from typing import Dict,Any

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
    parent_dir = os.path.dirname(module_dir)

    # data/ relative to parent directory (project root/data/)
    candidates.append(os.path.join(parent_dir, "GAAFpy", "rsm_coefficients.json"))

    # relative to current working directory
    candidates.append(os.path.join(os.getcwd(), "rsm_coefficients.json"))
    candidates.append(r"rsm_coefficients.json")  # Just in case

    # Try each candidate location
    for coeffs_file in candidates:
        if os.path.exists(coeffs_file):
            try:
                with open(coeffs_file, 'r') as f:
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