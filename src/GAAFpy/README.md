# The General Aviation Aircraft Family python package (GAAFpy) modules

This folder contains the different modules of the GAAFpy package. Variant and Family methods are separated, together with utility methods, in class-based modules. The resonse surface models are defined using coefficients which are loaded form the 'rsm_coefficients.json' file.

## Module Structure

```batch
.
├── __init__.py
├── family.py
├── README.md
├── rsm_coefficients.json
├── utils.py
└── variant.py
```

## Module Descriptions

- *family.py*: Contains the full GAABenchmark class implementation (vectorised evaluation, objective and constraint calculation). Specific GAA problem formulations can be constructed by filtering the objective and constraint outputs of the GAABenchmark.
- *utils.py*: Defines several utilities for the GAAFpy package, including design variable bounds and rsm coefficient loading/caching.
- *variant.py*: Contains an AircraftVariant class representation of an individual variant within the GAA problem.
