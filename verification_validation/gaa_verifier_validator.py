#-*- coding: utf-8 -*-
"""
gaa_verifier_validator.py
=================================

Description
-----------
Verification & Validation script for the Python implementation of the
General Aviation Aircraft (GAA) Benchmark Problem.

Module requires a CSV input file containing design vectors and their objective
and constraint values to compare against. This file can be constructed using
output data from e.g. the MOEA RealWorldBenchmark problem Java GAA
implementation.

Classes
-------
GAA_Validator
    Main class to perform validation of Python GAA implementation against
    reference data.

Examples
--------
>>>from gaa_verifier_validator import run_validation
>>>objs, cons, report = run_validation("path/to/csv",
                                       "path/to/report.txt")

References
----------
    [1] Zatarain Salazar, J., Hadka, D., Reed, P., Seada, H., and Deb, K.
            (2024). Diagnostic benchmarking of many-objective evolutionary
            algorithms for real-world problems. Engineering Optimization,
            1-22. https://doi.org/10.1080/0305215X.2024.2381818.

Versioning
----------
@author: T.S. Vermeulen
@email: T.S.Vermeulen@tudelft.nl
@version: 1.0
@date (dd-mm-yyyy): 19-03-2026

Changelog:
- V1.0: Initial version.
"""

# Import standard libraries
import sys
import csv
from pathlib import Path
from typing import Tuple, Dict

# Import 3rd party libraries
import numpy as np

# Module Constants
__author__ = "Thomas Stephan Vermeulen"
__copyright__ = "Copyright 2026, all rights reserved"
__status__ = "Release"


def add_src_to_path():
    """Add src directory to Python path to import gaa module."""
    script_dir = Path(__file__).parent.parent
    src_dir = script_dir / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

# Import GAABenchmark after path is configured
add_src_to_path()
from gaa import GAABenchmark # type: ignore


# Constants
OBJECTIVE_NAMES = ["max_NOISE",
                   "max_WEMP",
                   "max_DOC",
                   "max_ROUGH",
                   "max_WFUEL",
                   "max_PURCH",
                   "min_RANGE",
                   "min_LDMAX",
                   "min_VCMAX",
                   "PFPF"]

DESIGN_VAR_NAMES = ["CSPD2", "AR2", "SWEEP2", "DPROP2", "WINGLD2", "AF2",
                    "SEATW2", "ELODT2", "TAPER2", "CSPD4", "AR4", "SWEEP4",
                    "DPROP4", "WINGLD4", "AF4", "SEATW4", "ELODT4", "TAPER4",
                    "CSPD6", "AR6", "SWEEP6", "DPROP6", "WINGLD6", "AF6",
                    "SEATW6", "ELODT6", "TAPER6"]

# Expected number of design variables and objectives
N_DESIGN_VARS = 27
N_OBJECTIVES = 10
N_CONSTRAINTS = 18

# Set these paths to run validation
DEFAULT_CSV_PATH = r"verification_validation/MOEA-GAA-output.csv"  # Relative or absolute path
DEFAULT_REPORT_PATH = r"verification_validation/validation_report.txt"  # Optional


# Main Validator Class
class GAA_Validator:
    """Validator for GAA benchmark implementation against reference data."""

    def __init__(self,
                 csv_path: str) -> None:
        """
        Initialise validator.

        Args:
            csv_path: Path to CSV file with reference data
        """
        self.csv_path = csv_path


    def load_csv(self) -> bool:
        """
        Load reference data from CSV file.

        Returns:
            True if successful, False otherwise
        """
        try:
            with open(self.csv_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            if not rows:
                print(f"CSV file is empty: {self.csv_path}")
                return False

            self.n_solutions = len(rows)

            # Extract design variables (columns 0-26)
            self.design_variables = np.zeros((self.n_solutions, N_DESIGN_VARS))
            for i, row in enumerate(rows):
                for j, var_name in enumerate(DESIGN_VAR_NAMES):
                    self.design_variables[i, j] = float(row[var_name])

            # Extract objectives (columns 27-36)
            self.reference_objectives = np.zeros((self.n_solutions, N_OBJECTIVES))
            for i, row in enumerate(rows):
                for j, obj_name in enumerate(OBJECTIVE_NAMES):
                    self.reference_objectives[i, j] = float(row[obj_name])

            # Extract constraints (starting from column 37: Constr1, Constr2, ...)
            # Note: CSV should have Constr1 through Constr18
            self.reference_constraints = np.zeros((self.n_solutions, N_CONSTRAINTS))
            for i, row in enumerate(rows):
                for j in range(N_CONSTRAINTS):
                    constr_name = f"Constr{j+1}"
                    if constr_name in row:
                        self.reference_constraints[i, j] = float(row[constr_name])

            print(f"Loaded CSV with {self.n_solutions} solutions")
            print(f"Design variables: {self.design_variables.shape}")
            print(f"Reference objectives: {self.reference_objectives.shape}")
            print(f"Reference constraints: {self.reference_constraints.shape}")
            return True

        except FileNotFoundError:
            print(f"CSV file not found: {self.csv_path}")
            return False
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return False


    def evaluate_python_implementation(self) -> bool:
        """
        Evaluate using Python GAA implementation.

        Returns:
            True if successful, False otherwise
        """

        try:
            gaa = GAABenchmark(self.design_variables)
            (self.python_objectives, 
             self.python_constraints, 
             self.python_summed_cv) = gaa.evaluate()

            print(f"Python objectives: {self.python_objectives.shape}")
            print(f"Python constraints: {self.python_constraints.shape}")
            print(f"Python summed CV: {self.python_summed_cv.shape}")
            return True
        except Exception as e:
            print(f"Error in Python evaluation: {e}")
            import traceback
            traceback.print_exc()
            return False


    def compute_differences(self) -> Dict[str, np.ndarray]:
        """
        Compute differences between Python and reference results.

        Returns:
            Dictionary with objective and constraint differences
        """
        obj_diff = np.divide(self.python_objectives - self.reference_objectives,
                             self.reference_objectives,
                             where=self.reference_objectives!=0,
                             out=np.abs(self.python_objectives - self.reference_objectives))
        con_diff = np.divide(self.python_constraints - self.reference_constraints,
                             self.reference_constraints,
                             where=self.reference_constraints!=0,
                             out=np.abs(self.python_constraints - self.reference_constraints))

        return {"objectives_diff": obj_diff,
                "objectives_abs_diff": np.abs(obj_diff),
                "constraints_diff": con_diff,
                "constraints_abs_diff": np.abs(con_diff)}


    def generate_validation_report(self,
                                   output_path: str,
                                   obj_tolerance: float = 5e-6,
                                   con_tolerance: float = 5e-6) -> str:
        """
        Generate validation (txt) report.

        Args:
            output_path: Path to save report
            obj_tolerance: Tolerance for objective validation
            con_tolerance: Tolerance for constraint validation

        Returns:
            Report string
        """

        if not hasattr(self, 'python_objectives') or self.python_objectives is None:
            return "No results to report. Run evaluate_python_implementation() first."

        diffs = self.compute_differences()
        obj_diff = diffs["objectives_abs_diff"]
        con_diff = diffs["constraints_abs_diff"]

        # Compute statistics
        obj_max_diff = np.max(obj_diff)
        obj_mean_diff = np.mean(obj_diff)
        obj_std_diff = np.std(obj_diff)
        con_max_diff = np.max(con_diff)
        con_mean_diff = np.mean(con_diff)
        con_std_diff = np.std(con_diff)

        obj_passes = np.all(obj_diff < obj_tolerance)
        con_passes = np.all(con_diff < con_tolerance)

        # Build report
        report = []
        report.append("=" * 80)
        report.append("GAA BENCHMARK VERIFICATION/VALIDATION REPORT")
        report.append("=" * 80)
        report.append("")

        report.append("SUMMARY")
        report.append("-" * 80)
        report.append(f"Total solutions evaluated: {self.n_solutions}")
        report.append(f"CSV file: {self.csv_path}")
        report.append("")

        report.append("OBJECTIVES VALIDATION")
        report.append("-" * 80)
        report.append(f"Tolerance threshold: {obj_tolerance}")
        report.append(f"Status: {'PASS' if obj_passes else 'FAIL'}")
        report.append(f"Max relative difference: {obj_max_diff:.2e}")
        report.append(f"Mean relative difference: {obj_mean_diff:.2e}")
        report.append(f"Std deviation of differences: {obj_std_diff:.2e}")
        report.append("")

        report.append("CONSTRAINTS VALIDATION")
        report.append("-" * 80)
        report.append(f"Tolerance threshold: {con_tolerance}")
        report.append(f"Status: {'PASS' if con_passes else 'FAIL'}")
        report.append(f"Max relative difference: {con_max_diff:.2e}")
        report.append(f"Mean relative difference: {con_mean_diff:.2e}")
        report.append(f"Std deviation of differences: {con_std_diff:.2e}")
        report.append("")

        # Detailed objective statistics
        report.append("OBJECTIVES - DETAILED STATISTICS")
        report.append("-" * 80)
        for i, obj_name in enumerate(OBJECTIVE_NAMES):
            max_diff = np.max(obj_diff[:, i])
            mean_diff = np.mean(obj_diff[:, i])
            violations = np.sum(obj_diff[:, i] > obj_tolerance)
            status = "OK" if max_diff < obj_tolerance else "X "
            report.append(
                f"{status} {obj_name:12} | "
                f"Max: {max_diff:.2e} | "
                f"Mean: {mean_diff:.2e} | "
                f"Violations: {violations}/{self.n_solutions}"
            )
        report.append("")

        # Detailed constraint statistics
        report.append("CONSTRAINTS - DETAILED STATISTICS")
        report.append("-" * 80)
        for i in range(N_CONSTRAINTS):
            max_diff = np.max(con_diff[:, i])
            mean_diff = np.mean(con_diff[:, i])
            violations = np.sum(con_diff[:, i] > con_tolerance)
            status = "OK" if max_diff < con_tolerance else "X "
            report.append(
                f"{status} Constr{i+1:2} | "
                f"Max: {max_diff:.2e} | "
                f"Mean: {mean_diff:.2e} | "
                f"Violations: {violations}/{self.n_solutions}"
            )
        report.append("")

        # Solutions with largest errors
        report.append("SOLUTIONS WITH LARGEST ERRORS")
        report.append("-" * 80)

        # Find solutions with max objective error
        obj_error_sum = np.sum(obj_diff, axis=1)
        worst_obj_idx = np.argsort(obj_error_sum)[-5:][::-1]

        report.append("Top 5 solutions with largest objective errors:")
        for rank, idx in enumerate(worst_obj_idx, 1):
            if obj_error_sum[idx] > obj_tolerance:
                report.append(
                    f"  {rank}. Solution {idx}: Total error = {obj_error_sum[idx]:.2e}"
                )
        report.append("")

        # Find solutions with max constraint error
        con_error_sum = np.sum(con_diff, axis=1)
        worst_con_idx = np.argsort(con_error_sum)[-5:][::-1]

        report.append("Top 5 solutions with largest constraint errors:")
        for rank, idx in enumerate(worst_con_idx, 1):
            if con_error_sum[idx] > 0:
                report.append(
                    f"  {rank}. Solution {idx}: Total error = {con_error_sum[idx]:.2e}"
                )
        report.append("")

        report.append("=" * 80)
        report.append("END OF REPORT")
        report.append("=" * 80)

        report_str = "\n".join(report)

        # Save if path provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_str)
            print(f"\nReport saved to: {output_path}")

        return report_str


def run_validation(csv_path: str,
                   report_path: str) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Run validation.

    Args:
        csv_path: Path to reference CSV file with design solutions and results
        report_path: Path to save validation report

    Returns:
        Tuple of (python_objectives, python_constraints, report_string)

    Example:
        >>> csv = "verification_validation/MOEA-GAA-output.csv"
        >>> objs, cons, report = run_validation(csv, "report.txt")
        >>> print(report)
    """

    # Create validator
    validator = GAA_Validator(csv_path)

    # Load reference data
    print(f"Loading CSV from: {csv_path}")
    if not validator.load_csv():
        raise ValueError(f"Could not load CSV from {csv_path}")

    # Evaluate Python implementation
    print("\nEvaluating Python GAA implementation...")
    if not validator.evaluate_python_implementation():
        raise RuntimeError("Error during Python evaluation")

    # Generate report
    print("\nGenerating validation report...\n")
    report = validator.generate_validation_report(report_path)
    print(report)

    return validator.python_objectives, validator.python_constraints, report


if __name__ == "__main__":
    # Run validation with default paths configured above
    run_validation(csv_path=DEFAULT_CSV_PATH,
                   report_path=DEFAULT_REPORT_PATH)
