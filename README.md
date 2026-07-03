# Poroelastic phase-field model for fluid-driven fracture

This repository contains the finite element implementations accompanying the paper:

**A finite element poroelastic phase-field formulation for fluid-driven fracture**

The codes implement a two-dimensional plane-strain Darcy--Biot phase-field fracture model for fluid-driven fracture in porous media. The formulation couples small-strain Biot poroelasticity, Darcy flow, AT2 phase-field fracture, and a scalar damage-dependent permeability.

The repository provides two implementations of the same baseline example:

```text
firedrake/main.py   Firedrake implementation
fenics/main.py      FEniCS/DOLFIN implementation
````

The Firedrake implementation is the primary implementation used for the paper. The FEniCS/DOLFIN implementation is provided as a companion implementation of the same formulation.

## Model features

The implementation includes:

* small-strain Biot poroelasticity;
* Darcy flow with damage-dependent permeability;
* AT2 phase-field fracture regularisation;
* Amor split within a hybrid formulation;
* irreversibility enforced by projection;
* Gaussian regularised fluid injection at the notch centre.

The model deliberately does not use aperture-based Reynolds--Poiseuille flow, fracture-flow unknowns, or explicit crack-surface flow equations.

## Problem description

The default example is a square two-dimensional domain containing a horizontal pre-existing notch at the centre. Fluid is injected at the notch centre through a regularised Gaussian source. All external boundaries are mechanically fixed and hydraulically drained.

For the default mesh,

```text
mx = 200
my = 201
```

the expected problem size is approximately:

```text
Number of elements: 40200
Coupled displacement--pressure solve: 363808 DOFs
Phase-field solve: 40602 DOFs
Total reported DOFs: 404410
```

## Numerical note

The Gaussian source is narrow compared with the domain size. In the FEniCS/DOLFIN implementation, an explicit quadrature degree of 10 is used for domain integrals. This improves integration of the regularised source and gives pressure and fracture responses consistent with the Firedrake implementation.

## Running the Firedrake code

From the repository root, run:

```bash
cd firedrake
python main.py
```

The Firedrake code writes the displacement, pore pressure, and phase field to:

```text
solution.pvd
```

and writes the centre-pressure history to:

```text
p_c.txt
```

## Running the FEniCS/DOLFIN code

From the repository root, run:

```bash
cd fenics
python main.py
```

The FEniCS/DOLFIN code writes the displacement, pore pressure, and phase field to:

```text
results/solution_u.pvd
results/solution_p.pvd
results/solution_phi.pvd
```

and writes the centre-pressure history to:

```text
p_c.txt
```

## Dependencies

The Firedrake code requires a working Firedrake installation.

The FEniCS code requires legacy FEniCS/DOLFIN.

Both codes also use:

```text
numpy
matplotlib
```

Firedrake and/or FEniCS should be installed following their official installation instructions.

## Notes

This is research code intended to reproduce the baseline numerical example from the paper. It is kept as a compact, readable script.

Small numerical differences may still occur because of differences in finite element backends, linear solvers, quadrature implementation, and output handling.

## Author

Masoud Ahmadi,
Department of Mathematics,
University College London,
2026
