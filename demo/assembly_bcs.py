# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# Copyright (C) 2022 Jørgen Schartum Dokken
#
# This file is part of Oasisx
# SPDX-License-Identifier:    MIT
#
# # Application of Dirichlet BCs
# Illustrates how to apply symmetric bcs for the RHS of the problem when using matrix caching
#
# We start by importing the necessary modules

import time
import seaborn
import pandas
import dolfinx
from mpi4py import MPI
from petsc4py import PETSc
import ufl
import numpy as np


def assembly(mesh, P: int, repeats: int, jit_options: dict = None):
    V = dolfinx.fem.FunctionSpace(mesh, ("CG", int(P)))

    def f(x):
        return 2*np.sin(x[0])+3+2*x[1]
    mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    boundary_dofs = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim-1, boundary_facets)
    g = dolfinx.fem.Function(V)
    g.interpolate(f)
    bcs = [dolfinx.fem.dirichletbc(g, boundary_dofs)]

    dt = 0.5
    nu = 0.3
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Solution from previous time step
    u_1 = dolfinx.fem.Function(V)
    u_1.interpolate(lambda x: np.sin(x[0])*np.cos(x[1]))

    # Define variational forms
    mass = ufl.inner(u, v) * ufl.dx
    stiffness = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx

    u_ab = [dolfinx.fem.Function(V, name=f"u_ab{i}") for i in range(mesh.geometry.dim)]
    convection = ufl.inner(ufl.dot(ufl.as_vector(u_ab), ufl.nabla_grad(u)), v) * ufl.dx
    for u_abi in u_ab:
        u_abi.interpolate(lambda x: x[0])

    # Compile forms for matrix vector products
    jit_options = {} if jit_options is None else jit_options
    mass_form = dolfinx.fem.form(mass, jit_params=jit_options)
    stiffness_form = dolfinx.fem.form(stiffness, jit_params=jit_options)
    convection_form = dolfinx.fem.form(convection, jit_params=jit_options)

    # Compile form for vector assembly (action)
    dt_inv = dolfinx.fem.Constant(mesh, 1./dt)
    dt_inv.name = "dt_inv"
    nu_c = dolfinx.fem.Constant(mesh, nu)
    nu_c.name = "nu"
    lhs = dt_inv * mass - 0.5 * nu_c * stiffness - 0.5*convection
    rhs = dolfinx.fem.form(ufl.action(lhs, u_1), jit_params=jit_options)
    lhs_form = dolfinx.fem.form(lhs)

    # Assemble time independent matrices
    # Mass matrix
    M = dolfinx.fem.petsc.create_matrix(mass_form)
    M.setOption(PETSc.Mat.Option.SYMMETRY_ETERNAL, True)
    M.setOption(PETSc.Mat.Option.IGNORE_ZERO_ENTRIES, True)
    dolfinx.fem.petsc.assemble_matrix(M, mass_form)
    M.assemble()
    M.setOption(PETSc.Mat.Option.NEW_NONZERO_LOCATIONS, False)

    # Stiffness matrix
    K = dolfinx.fem.petsc.create_matrix(stiffness_form)
    K.setOption(PETSc.Mat.Option.SYMMETRY_ETERNAL, True)
    K.setOption(PETSc.Mat.Option.IGNORE_ZERO_ENTRIES, True)
    dolfinx.fem.petsc.assemble_matrix(K, stiffness_form)
    K.assemble()
    K.setOption(PETSc.Mat.Option.NEW_NONZERO_LOCATIONS, False)

    # Create time dependent matrix (convection matrix)
    A = dolfinx.fem.petsc.create_matrix(mass_form)
    A.setOption(PETSc.Mat.Option.IGNORE_ZERO_ENTRIES, True)

    # Vector for matrix vector product
    b = dolfinx.fem.Function(V)

    t_matvec = np.zeros((repeats, mesh.comm.size), dtype=np.float64)
    t_action = np.zeros((repeats, mesh.comm.size), dtype=np.float64)
    t_caction = np.zeros((repeats, mesh.comm.size), dtype=np.float64)

    for i in range(repeats):
        # Zero out time-dependent matrix
        A.zeroEntries()

        # Add convection term
        dolfinx.fem.petsc.assemble_matrix(A, convection_form)
        A.assemble()

        # Do mat-vec operations
        start_matvec = time.perf_counter()
        A.scale(-0.5)
        A.axpy(1./dt, M)
        A.axpy(-0.5*nu, K)
        dolfinx.cpp.fem.petsc.insert_diagonal(A, V._cpp_object, bcs=bcs, diagonal=1)
        A.assemble()
        A.mult(u_1.vector, b.vector)
        t_lifting = time.perf_counter()

        dolfinx.fem.petsc.apply_lifting(b.vector, [mass_form], [bcs], scale=1./dt)
        dolfinx.fem.petsc.apply_lifting(
            b.vector, [stiffness_form], [bcs], scale=-0.5*nu)
        dolfinx.fem.petsc.apply_lifting(
            b.vector, [convection_form], [bcs], scale=-0.5)
        t_l2 = time.perf_counter()
        b.x.scatter_reverse(dolfinx.la.ScatterMode.add)
        dolfinx.fem.petsc.set_bc(b.vector, bcs)
        b.x.scatter_forward()
        end_matvec = time.perf_counter()
        matvec = end_matvec - start_matvec

        print(
            f"Matvec lifting {t_l2 - t_lifting:3e}, matvec total {matvec:3e}, fraction lift {(t_l2-t_lifting)/(matvec):3f}")

        # Compute the vector without using pre-generated matrices
        b_d = dolfinx.fem.Function(V)
        start_action = time.perf_counter()
        dolfinx.fem.petsc.assemble_vector(b_d.vector, rhs)
        t_lifting = time.perf_counter()
        dolfinx.fem.petsc.apply_lifting(b_d.vector, [mass_form], [bcs], scale=1./dt)
        dolfinx.fem.petsc.apply_lifting(b_d.vector, [stiffness_form], [bcs], scale=-0.5*nu)
        dolfinx.fem.petsc.apply_lifting(b_d.vector, [convection_form], [bcs], scale=-0.5)
        t_l2 = time.perf_counter()
        b_d.x.scatter_reverse(dolfinx.la.ScatterMode.add)
        dolfinx.fem.petsc.set_bc(b_d.vector, bcs)
        b_d.x.scatter_forward()
        end_action = time.perf_counter()
        action = end_action - start_action
        print(
            f"Action lifting {t_l2 - t_lifting:3e}, Action total {action:3e}, fraction lift {(t_l2-t_lifting)/action:3f}")
        # Apply lifiting with the combined form
        b2 = dolfinx.fem.Function(V)
        start_combined = time.perf_counter()
        dolfinx.fem.petsc.assemble_vector(b2.vector, rhs)
        t_lifting = time.perf_counter()
        dolfinx.fem.petsc.apply_lifting(b2.vector, [lhs_form], [bcs])
        t_l2 = time.perf_counter()
        b2.x.scatter_reverse(dolfinx.la.ScatterMode.add)
        dolfinx.fem.petsc.set_bc(b2.vector, bcs)
        end_combined = time.perf_counter()
        combined = end_combined - start_combined
        print(
            f"Combined lifting {t_l2 - t_lifting:3e}, Combined total {end_combined-start_combined:3e}, fraction lift {(t_l2-t_lifting)/(end_combined-start_combined):3f}")

        # Compare results
        assert np.allclose(b_d.x.array, b2.x.array)
        assert np.allclose(b.x.array, b_d.x.array)

        # Get timings

        t_matvec[i, :] = mesh.comm.allgather(matvec)
        t_action[i, :] = mesh.comm.allgather(action)
        t_caction[i, :] = mesh.comm.allgather(combined)
    return V.dofmap.index_map_bs * V.dofmap.index_map.size_global, t_matvec, t_action, t_caction


# We solve the problem on a unit cube that is split into tetrahedras with `Nx`,`Ny` and `Nx`
# tetrahera in the x, y and z-direction respectively.

def run_parameter_sweep(Nx: int, Ny: int, Nz: int, repeats: int, min_degree: int,
                        max_degree: int) -> dict:
    # Information regarding optimization flags can be found at:
    # https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html
    jit_options = {"cffi_extra_compile_args": ["-march=native", "-O3"]}

    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, Nx, Ny, Nz)
    Ps = np.arange(min_degree, max_degree+1, dtype=np.int32)
    j = 0
    results = {}
    for i, P in enumerate(Ps):
        dof, matvec, action, caction = assembly(mesh, P, repeats=repeats, jit_options=jit_options)
        for row in matvec:
            for process in row:
                results[j] = {"P": P, "num_dofs": dof, "method":
                              "matvec", "time (s)": process, "procs": MPI.COMM_WORLD.size}
                j += 1
        for row in action:
            for process in row:
                results[j] = {"P": P, "num_dofs": dof, "method":
                              "action", "time (s)": process, "procs": MPI.COMM_WORLD.size}
                j += 1
        for row in action:
            for process in row:
                results[j] = {"P": P, "num_dofs": dof, "method":
                              "caction", "time (s)": process, "procs": MPI.COMM_WORLD.size}
                j += 1
    return results


# We use `pandas` and `seaborn` to visualize the results

# +

def create_plot(results: dict, outfile: str):
    if MPI.COMM_WORLD.rank == 0:
        df = pandas.DataFrame.from_dict(results, orient="index")
        df["label"] = "P" + df["P"].astype(str) + " " + \
            df["num_dofs"].astype(str) + " \n Comms: " + df["procs"].astype(str)
        plot = seaborn.catplot(data=df, kind="swarm",  x="label", y="time (s)", hue="method")
        plot.set(yscale="log")
        import matplotlib.pyplot as plt
        plt.grid()
        plt.savefig(outfile)


# -

# We start by running the comparison for an increasing number of degrees of freedom on a fixed grid.

if __name__ == "__main__":
    results_p = run_parameter_sweep(35, 35, 35, repeats=3, min_degree=1, max_degree=3)
    create_plot(results_p, "P_sweep_bc.png")

# We observe that for all the runs, independent of the degree $P$, the *Action Strategy* is
# significantly faster than the

# We note that the run for $P=1$ is relatively small, and therefore run $P=1$ on a larger mesh

# if __name__ == "__main__":
#     results_p1 = run_parameter_sweep(100, 100, 80, 3, 1, 1)
#     create_plot(results_p1, "P1.png")

# We observe that the run-time of both strategies for $P=1$ are more or less the
# same for larger matrices.
