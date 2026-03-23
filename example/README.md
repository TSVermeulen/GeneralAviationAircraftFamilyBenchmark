# Example usage of GAAFpy

This folder contains two examples to use the GAAFpy package in an optimisation context to optimise both the complete GAA family,
and each aircraft individually. Both examples use the Pymoo package, which can be installed using:

```console
pip install -U pymoo
```

Note that the family optimisation example is the same as that described in the main repository readme.
For the individual aircraft optimisation, the aircraft type being optimised is set using the VARIANT_TYPE integer ([0,1,2]) at the top of the __main__ section.
