#############################################
#         Poroelasticity fracture           # 
#############################################
"""
Darcy--Biot phase-field fracture
The model uses small-strain Biot poroelasticity, 
AT2 phase-field fracture, and a scalar
damage-dependent permeability.

Written by: Masoud Ahmadi
University College London,
Department of Mathematics,
2026
"""

# ====================
# Import libraries
# ====================
from fenics import *
from ufl import max_value, min_value
import matplotlib.pyplot as plt
import numpy as np
import os

# Time
import time
start_time = time.time()


# ====================
# Input parameters
# ====================

# Material & Geometry parameters
E     = Constant(10.0e9)   # Young's modulus (Pa)
nu    = Constant(0.3)      # Poisson's ratio
alpha = Constant(1.0)      # Biot's coefficient
S     = Constant(1.0e-10)  # Storage coefficient (Pa^-1) [S = 1/M where M: Biot's modulus]
k_0   = Constant(1.0e-15)  # Matrix permeability (m^2) [of intact material]
k_1   = Constant(1.0e-13)  # Permeability of fractured material (m^2)
n     = Constant(2.0)      # Exponent for permeability function
mu_vis= Constant(1.0e-3)   # Dynamic viscosity (Pa.s)
Gc    = Constant(100)      # Critical energy release rate (N/m)
ell   = Constant(0.015)    # Regularization length (m)
L_1   = 1.0                # Length in x-direction (m)
L_2   = 1.0                # Length in y-direction (m)
# L_3   = 1.0              # Length in z-direction (m) [for 3D]
a_0   = 0.1                # Length of the initial crack (m)

# Time parameters
dt      = 0.002   # Time step size (s)
t_end   = 4.0     # Final time (s)
n_steps = int(t_end/dt) # Number of time steps

# Output parameters
output_interval = 20     # Every 'output_interval' time steps, results will be saved to VTK
live_plot       = True   # Whether to display live plots

# Initial conditions and source term
p_0  = Constant(0.0)  # Initial pore pressure
Q = Constant(0.0001)  # Injection rate

# Output directory
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

# ====================
# Mesh
# ====================

## Create a uniform mesh
mx = 200     # Number of elements in x-direction
my = mx + 1  # Number of elements in y-direction
# mz = mx    # Number of elements in z-direction [for 3D]

# 2D mesh
mesh = RectangleMesh.create([Point(0.0, 0.0), Point(L_1, L_2)], [mx, my], CellType.Type.quadrilateral)
x, y = SpatialCoordinate(mesh)

# Higher quadrature degree for integration
dx = Measure("dx", domain=mesh, metadata={"quadrature_degree": 10})  # [warning]: without this, it does not match the Firedrake results 

# Minimum mesh size
min_dy = L_2 / my  # Minimum mesh size in y-direction


# ====================
# FE settings
# ====================

# Function spaces
V_el  = VectorElement("CG", mesh.ufl_cell(), 2)  # Quadratic for displacement
W_el  = FiniteElement("CG", mesh.ufl_cell(), 1)  # Linear for pore pressure
X_el  = FiniteElement("CG", mesh.ufl_cell(), 1)  # Linear for phase field
V     = FunctionSpace(mesh, V_el)                # Quadratic for displacement
W     = FunctionSpace(mesh, W_el)                # Linear for pore pressure
X     = FunctionSpace(mesh, X_el)                # Linear for phase field
Z     = FunctionSpace(mesh, MixedElement([V_el, W_el])) # Mixed function space for u, p
V_out = VectorFunctionSpace(mesh, "CG", 1)       # Piecewise linear for output

# Trial and Test functions
z         = Function(Z)      # (u, p)
u, p      = split(z)         # Extract u, p
phi       = Function(X)      # Phase field
z_old     = Function(Z)      # Old (u, p)
u_old, p_old = split(z_old)      # Extract old u, p
phi_old      = Function(X)       # Old phi
del_u, del_p = TestFunctions(Z)  # Test functions
del_phi      = TestFunction(X)   # Test function for phi
du, dp       = TrialFunctions(Z) # Trial functions for (u, p)
dphi         = TrialFunction(X)  # Trial function for phi


# ====================
#  Model setup 
# ====================

## Elasticity parameters
lmbda   = E*nu / ((1 + nu)*(1 - 2*nu))  # First Lame parameter
mu      = E / (2*(1 + nu))              # Second Lame parameter (shear modulus)
dim     = mesh.geometric_dimension()    # Dimension of the problem
K_bulk  = lmbda + 2*mu/3.0              # Bulk modulus


## Strain and Stress tensors
I = Identity(dim)  # Identity tensor

def epsilon(u):
    return sym(grad(u))

def sigma_eff(u):
    return lmbda*div(u)*I + 2*mu*epsilon(u)

def sigma(u, p):
    return g_phi*sigma_eff(u) - alpha*p*I


## g_phi function
eta     = Constant(1e-12)  # To avoid stiffness degeneracy condition
g_phi   = (1.0 - phi)**2 + eta


## Positive part of the elastic energy density function
# Macaulay bracket
def Macaulay(x):
    return conditional(gt(x, 0), x, 0)

def psi_p(u):
    psi_p_vol = 0.5 * K_bulk * (Macaulay(tr(epsilon(u))))**2
    if dim ==2:
        # plane strain:
        psi_dev = mu * (inner(epsilon(u), epsilon(u)) - (1.0/3.0)*tr(epsilon(u))**2)
    
    elif dim ==3:
        psi_dev = mu * inner(dev(epsilon(u)), dev(epsilon(u)))

    return psi_p_vol + psi_dev


## Permeability function
def k(phi):
    return k_0 + (k_1 - k_0) * phi**n

## Define zeta
def zeta(u, p):
    return S*p + alpha*div(u)


# ====================
#  Boundary conditions
# ====================

# Dirichlet boundary conditions for displacement
bcs_u = [DirichletBC(Z.sub(0), Constant((0.0, 0.0)), "on_boundary")]
# bcs_u = [DirichletBC(Z.sub(0), Constant((0.0, 0.0, 0.0)), "on_boundary")] # for 3D

# Dirichlet boundary conditions for pore pressure
bcs_p = [DirichletBC(Z.sub(1), Constant(0.0), "on_boundary")]
         
# Combine boundary conditions
bcs = bcs_u + bcs_p

## Gaussian source term
eps = min_dy / 2 
r2 = (x - L_1/2)**2 + (y - L_2/2)**2
gaussian = exp(-r2 / eps**2)
gaussian_integral = assemble(gaussian * dx)
Q_gaus = Q * gaussian / gaussian_integral

# Check total injection rate: 
Q_total = assemble(Q_gaus * dx)
Injection_error = abs(float((Q_total - Q) / Q * 100))
print(f"Injected rate (numerical) = {Q_total:.4e} m^2/s | Target Q = {float(Q):.4e} m^2/s | Error = {Injection_error:.3f} %")


# =====================
#  Initial conditions
# =====================

## Initial pore pressure
z.vector()[:] = 0.0
z_old.assign(z)

# Initial crack(s)
phi_0 = interpolate(Expression("(x[0] <= (L_1+a_0)/2.0 && x[0] >= (L_1-a_0)/2.0 && x[1] < 0.5*L_2+min_dy && x[1] > 0.5*L_2-min_dy) ? 1.0 : 0.0", degree=1, L_1=L_1, L_2=L_2, a_0=a_0, min_dy=min_dy), X)
phi.assign(phi_0)
phi_old.assign(phi_0)

coords = V.tabulate_dof_coordinates()  # or mesh.coordinates() for vertices
xs, ys = coords[:,0], coords[:,1]

# for every point, check a mirrored point exists
import numpy as np
for i in range(0, len(xs), 500):  # sample
    x0, y0 = xs[i], ys[i]
    mirror_x = L_1 - x0
    match = np.any((np.isclose(xs, mirror_x, atol=1e-12)) & (np.isclose(ys, y0, atol=1e-12)))
    if not match:
        print("no mirror for", x0, y0)


# ====================
#  Weak Form
# ====================

# 1 & 2. Linear momentum balance and Mass balance
a_up = (inner(epsilon(del_u), sigma(du, dp)) + del_p*(zeta(du, dp))/dt + (k(phi)/mu_vis)*dot(grad(del_p), grad(dp))) * dx
L_up = (del_p*(zeta(u_old, p_old))/dt + del_p*Q_gaus) * dx

# 3. Fracture field
a_phi = (Gc*ell*dot(grad(dphi), grad(del_phi)) + (Gc/ell)*dphi*del_phi + 2.0*psi_p(u)*dphi*del_phi) * dx
L_phi = 2.0*psi_p(u)*del_phi*dx


# =====================
# Solve the problem
# =====================

# Create the VTK file for output
outfile_u   = File(f"{output_dir}/solution_u.pvd")
outfile_p   = File(f"{output_dir}/solution_p.pvd")
outfile_phi = File(f"{output_dir}/solution_phi.pvd")

u_out = project(u, V_out)
p_out = project(p, W)
u_out.rename("u", "u")
p_out.rename("p", "p")
phi.rename("phi", "phi")
outfile_u   << (u_out, 0.0)
outfile_p   << (p_out, 0.0)
outfile_phi << (phi, 0.0)

# Print problem information
print("======================================")
print("Poroelasticity fracture simulation")
print("Number of elements: ", mesh.num_cells())
print(f"Number of DOFs:", Z.dim()+X.dim())
print("======================================")


## Time-stepping

# Set up solvers
up_problem = LinearVariationalProblem(a_up, L_up, z, bcs=bcs)
up_solver = LinearVariationalSolver(up_problem)
phi_problem = LinearVariationalProblem(a_phi, L_phi, phi)
phi_solver = LinearVariationalSolver(phi_problem)
up_solver.parameters["linear_solver"] = "mumps"
phi_solver.parameters["linear_solver"] = "mumps"

# Live plot setup
p_c = [0.0]
rtime = [0.0]
x_c, y_c = L_1/2, L_2/2
def centre_pressure_value(p_h):
    coords = p_h.function_space().tabulate_dof_coordinates().reshape((-1, dim))
    vals = p_h.vector().get_local()
    ids = np.where(np.isclose(coords[:, 0], x_c, atol=1.0e-12))[0]
    below = ids[coords[ids, 1] <= y_c]
    above = ids[coords[ids, 1] >= y_c]
    i0 = below[np.argmax(coords[below, 1])]
    i1 = above[np.argmin(coords[above, 1])]
    y0, y1 = coords[i0, 1], coords[i1, 1]
    if abs(y1 - y0) < 1.0e-14:
        return vals[i0]
    return vals[i0] + (vals[i1] - vals[i0])*(y_c - y0)/(y1 - y0)

if live_plot:
    plt.ion()
    ax = plt.gca()
    fig = plt.gcf()
    line, = ax.plot([], [], 'b-', linewidth=2)
    ax.set_xlabel('Time [s]', fontsize=12)
    ax.set_ylabel('p_c [MPa]', fontsize=12)
    ax.set_title('Centre pressure', fontsize=13)
    ax.grid(True, alpha=0.3)

# Time-stepping loop
for i in range(n_steps):

    # Solve the system of equations
    up_solver.solve()
    phi_solver.solve() 

    # Prevent healing
    phi.vector()[:] = np.maximum(phi_old.vector().get_local(), phi.vector().get_local())

    # Boundedness
    phi.vector()[:] = np.minimum(np.maximum(phi.vector().get_local(), 0.0), 1.0)
    
    # Update the old solutions
    z_old.assign(z)
    phi_old.assign(phi)
    
    # Save results
    if (i+1) % output_interval == 0:
        u_out = project(u, V_out)
        p_out = project(p, W)
        u_out.rename("u", "u")
        p_out.rename("p", "p")
        phi.rename("phi", "phi")
        outfile_u   << (u_out, dt*(i+1))
        outfile_p   << (p_out, dt*(i+1))
        outfile_phi << (phi, dt*(i+1))

    # Results Monitoring
    rtime.append(float(dt*(i+1)))
    p_h = z.sub(1, deepcopy=True)
    p_c_now = centre_pressure_value(p_h)
    p_c.append(p_c_now/1e6)
    if live_plot:
        line.set_xdata(rtime)
        line.set_ydata(p_c)
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()

    # Print progress
    print(f"time={float(dt)*float(i+1):.3f} | p_c={p_c[-1]:.3f} MPa | [{100.0*float(i+1)/float(n_steps):.2f}%]")


# Print completion message
print("======================================")
print("Simulation Time: {:.2f} s".format(time.time() - start_time))
print("Simulation completed successfully \U00002705")
print("======================================")

# Plot the final curve
if live_plot:
    plt.ioff()
    plt.show()

# Save the p_c data to a text file
np.savetxt(f"p_c.txt", np.column_stack([rtime, p_c]), header="time [s]    p_c [MPa]", fmt="%.8e")
