# Copyright (C) 2022 Jørgen Schartum Dokken
#
# This file is part of Oasisx
# SPDX-License-Identifier:    MIT
#
# This demo illustrates the performance differences between matrix free and cached assembly
# for the Crank Nicholson time discretization with the implicit Adams-Bashforth linearization
# in the tentative velocity step of the Navier-Stokes Equation


import dolfinx
from mpi4py import MPI
from petsc4py import PETSc
import ufl
import numpy as np


def assembly(mesh, P: int, repeats: int, jit_options: dict = None):
    V = dolfinx.fem.FunctionSpace(mesh, ("CG", int(P)))

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
    [u_abi.interpolate(lambda x: x[0]) for u_abi in u_ab]

    # Compile forms for matrix vector products
    mass_form = dolfinx.fem.form(mass, jit_params=jit_options)
    stiffness_form = dolfinx.fem.form(stiffness, jit_params=jit_options)
    convection_form = dolfinx.fem.form(convection, jit_params=jit_options)

    # Compile form for vector assembly (action)
    dt_inv = dolfinx.fem.Constant(mesh, 1./dt)
    dt_inv.name = "dt_inv"
    nu_c = dolfinx.fem.Constant(mesh, nu)
    nu_c.name = "nu"
    lhs = dt_inv * mass - 0.5 * nu_c * stiffness - 0.5*convection
    lhs = dolfinx.fem.form(ufl.action(lhs, u_1), jit_params=jit_options)

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

    for i in range(repeats):
        # Zero out time-dependent matrix
        A.zeroEntries()

        # Add convection term
        dolfinx.fem.petsc.assemble_matrix(A, convection_form)
        A.assemble()

        # Do mat-vec operations
        with dolfinx.common.Timer(f"~{P} {i} Mat-vec product") as _:
            A.scale(-0.5)
            A.axpy(1./dt, M)
            A.axpy(-0.5*nu, K)
            A.mult(u_1.vector, b.vector)
            b.x.scatter_forward()

        # Compute the vector without using pre-generated matrices
        b_d = dolfinx.fem.Function(V)
        with dolfinx.common.Timer(f"~{P} {i} Action") as _:
            dolfinx.fem.petsc.assemble_vector(b_d.vector, lhs)
            b_d.x.scatter_reverse(dolfinx.la.ScatterMode.add)
            b_d.x.scatter_forward()
        # Compare results
        assert np.allclose(b.x.array, b_d.x.array)

        # Get timings
        t_matvec = dolfinx.common.timing(f"~{P} {i} Mat-vec product")
        t_action = dolfinx.common.timing(f"~{P} {i} Action")
        t_matvec_all = mesh.comm.allgather(t_matvec[1])
        t_action_all = mesh.comm.allgather(t_action[1])

    return V.dofmap.index_map_bs * V.dofmap.index_map.size_global, t_matvec_all, t_action_all


if __name__ == "__main__":
    Nx, Ny, Nz = 50, 50, 50
    repeat = 3
    min_degree, max_degree = 1, 3
    figure = "output.ong"

    # Information regarding optimization flags can be found at:
    # https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html
    jit_options = {"cffi_extra_compile_args": ["-march=native", "-O3"]}

    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, Nx, Ny, Nz)
    Ps = np.arange(min_degree, max_degree+1, dtype=np.int32)
    num_dofs = np.zeros(len(Ps))
    t_matvec = np.zeros((len(Ps), mesh.comm.size), dtype=np.float64)
    t_action = np.zeros((len(Ps), mesh.comm.size), dtype=np.float64)
    results = {}
    j = 0
    for i, P in enumerate(Ps):
        dof, matvec, action = assembly(mesh, P, repeats=repeat, jit_options=jit_options)
        for t in matvec:
            results[j] = {"P": P, "num_dofs": dof, "method":
                          "matvec", "time": t, "procs": MPI.COMM_WORLD.size}
            j += 1
        for t in action:
            results[j] = {"P": P, "num_dofs": dof, "method":
                          "action", "time": t, "procs": MPI.COMM_WORLD.size}
            j += 1

    if MPI.COMM_WORLD.rank == 0:
        import pandas
        df = pandas.DataFrame.from_dict(results, orient="index")
        # df.to_csv(f"results.csv", mode="a")
        # df = pandas.read_csv(f"results.csv")
        import seaborn
        df["label"] = "P" + df["P"].astype(str) + " " + \
            df["num_dofs"].astype(str) + " \n Comms: " + df["procs"].astype(str)
        plot = seaborn.catplot(data=df, kind="swarm",  x="label", y="time", hue="method")
        plot.set(yscale="log")
        import matplotlib.pyplot as plt
        plt.grid()
        plt.savefig(figure)
