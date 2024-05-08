# Copyright (C) 2022 Jørgen Schartum Dokken
#
# This file is part of Oasisx
# SPDX-License-Identifier:    MIT

from typing import List, Optional, Tuple

from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
import numpy as np
import numpy.typing as npt
import pytest
import scipy.sparse
import ufl

from oasisx import DirichletBC, FractionalStep_AB_CN, LocatorMethod, PressureBC


def gather_PETScMatrix(
    A: PETSc.Mat,  # type: ignore
    comm: MPI.Comm,
    root: int = 0,
) -> scipy.sparse.csr_matrix:
    """
    Given a distributed PETSc matrix, gather in on process 'root' in
    a scipy CSR matrix
    """
    ai, aj, av = A.getValuesCSR()
    aj_all = comm.gather(aj, root=root)  # type: ignore
    av_all = comm.gather(av, root=root)  # type: ignore
    ai_all = comm.gather(ai, root=root)  # type: ignore
    if comm.rank == root:
        ai_cum = [0]
        for ai in ai_all:  # type: ignore
            offsets = ai[1:] + ai_cum[-1]
            ai_cum.extend(offsets)
        return scipy.sparse.csr_matrix(
            (np.hstack(av_all), np.hstack(aj_all), ai_cum),  # type: ignore
            shape=A.getSize(),
        )


def create_tentative_forms(
    mesh: dolfinx.mesh.Mesh,
    el_u: Tuple[str, int],
    el_p: Tuple[str, int],
    dt: float,
    nu: float,
    f: Optional[npt.NDArray[np.float64]],
) -> Tuple[
    ufl.Form,
    List[ufl.Form],
    dolfinx.fem.Function,
    dolfinx.fem.Function,
    dolfinx.fem.Function,
]:
    """
    Direct implementation of the i-th component of the tentative velocity equation
    """

    V = dolfinx.fem.functionspace(mesh, el_u)
    Q = dolfinx.fem.functionspace(mesh, el_p)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    p = dolfinx.fem.Function(Q)
    u_n = dolfinx.fem.Function(V)
    u_n2 = dolfinx.fem.Function(V)

    dx = ufl.Measure("dx", domain=mesh)
    u_ab = ufl.as_vector((1.5 * u_n - 0.5 * u_n2, 1.5 * u_n - 0.5 * u_n2))
    u_avg = 0.5 * (u + u_n)
    F = 1.0 / dt * (u - u_n) * v * dx
    F += ufl.dot(u_ab, ufl.grad(u_avg)) * v * dx
    F += nu * ufl.inner(ufl.grad(u_avg), ufl.grad(v)) * dx
    a, L = ufl.system(F)
    Ls = []
    for i in range(mesh.geometry.dim):
        if f is None:
            Ls.append(L + p * v.dx(i) * dx)
        else:
            Ls.append(L + p * v.dx(i) * dx + f[i] * v * dx)

    return a, Ls, u_n, u_n2, p


@pytest.mark.parametrize("body_force", [True, False])
@pytest.mark.parametrize("low_memory", [True, False])
def test_tentative(low_memory, body_force):
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)
    dim = mesh.topology.dim - 1
    el_u = ("Lagrange", 1)
    el_p = ("Lagrange", 1)

    solver_options = {"tentative": {"ksp_type": "preonly", "pc_type": "lu"}}
    options = {"low_memory_version": low_memory}

    if body_force:
        f = np.array([0.3, -0.1], dtype=np.float64)
    else:
        f = None

    def left_edge(x):
        return np.isclose(x[0], 0)

    def top_and_bottom(x):
        return np.logical_or(np.isclose(x[1], 0), np.isclose(x[1], 1))

    def outlet(x):
        return np.isclose(x[0], 1)

    # Locate facets for boundary conditions and create meshtags
    left_facets = dolfinx.mesh.locate_entities_boundary(mesh, dim, left_edge)
    left_value = 1
    tb_facets = dolfinx.mesh.locate_entities_boundary(mesh, dim, top_and_bottom)
    tb_value = 2
    right_facets = dolfinx.mesh.locate_entities_boundary(mesh, dim, outlet)
    right_value = 3
    facets = np.hstack([left_facets, tb_facets, right_facets])
    values = np.hstack(
        [
            np.full_like(left_facets, left_value, dtype=np.int32),
            np.full_like(tb_facets, tb_value, dtype=np.int32),
            np.full_like(right_facets, right_value, dtype=np.int32),
        ]
    )
    sort = np.argsort(facets)
    facet_tags = dolfinx.mesh.meshtags(mesh, dim, facets[sort], values[sort])

    # Create boundary conditions
    class Inlet:
        def __init__(self, t):
            self.t = t

        def eval(self, x):
            return (1 + self.t) * np.sin(np.pi * x[1])

    inlet = Inlet(0)

    bc_tb = DirichletBC(0.0, LocatorMethod.TOPOLOGICAL, (facet_tags, tb_value))
    bc_inlet_x = DirichletBC(inlet.eval, LocatorMethod.TOPOLOGICAL, (facet_tags, left_value))
    bc_inlet_y = DirichletBC(0.0, LocatorMethod.TOPOLOGICAL, (facet_tags, left_value))
    bcs_u = [[bc_inlet_x, bc_tb], [bc_inlet_y, bc_tb]]
    p_value = 4.0
    bcs_p = [PressureBC(p_value, (facet_tags, right_value))]

    # Create fractional step solver
    solver = FractionalStep_AB_CN(
        mesh,
        el_u,
        el_p,
        bcs_u=bcs_u,
        bcs_p=bcs_p,
        solver_options=solver_options,
        options=options,
        body_force=f,
    )

    dt = 0.1
    nu = 0.5

    # Set some almost sensible initial conditions
    inlet.t = -2 * dt
    solver._u2[0].interpolate(inlet.eval)
    solver._u2[1].interpolate(inlet.eval)
    inlet.t = -dt
    solver._u1[0].interpolate(inlet.eval)
    solver._u1[1].interpolate(inlet.eval)
    inlet.t = dt
    bc_inlet_x.update_bc()
    solver._ps.interpolate(lambda x: x[1])
    solver.assemble_first(dt, nu)
    solver.velocity_tentative_assemble()
    solver.velocity_tentative_solve()
    A_oasis = solver._A

    # Reference implementation
    a, Ls, u_n, u_n2, p = create_tentative_forms(mesh, el_u, el_p, dt, nu, f)
    V = u_n.function_space
    # Create bcs and boundary conditions
    p.interpolate(lambda x: x[1])
    ux = dolfinx.fem.Function(V)
    ux.interpolate(inlet.eval)
    inlet.t = -2 * dt
    u_n2.interpolate(inlet.eval)
    inlet.t = -dt
    u_n.interpolate(inlet.eval)
    bcs_u = [
        [
            dolfinx.fem.dirichletbc(ux, dolfinx.fem.locate_dofs_topological(V, dim, left_facets)),
            dolfinx.fem.dirichletbc(0.0, dolfinx.fem.locate_dofs_topological(V, dim, tb_facets), V),
        ],
        [
            dolfinx.fem.dirichletbc(
                0.0, dolfinx.fem.locate_dofs_topological(V, dim, left_facets), V
            ),
            dolfinx.fem.dirichletbc(0.0, dolfinx.fem.locate_dofs_topological(V, dim, tb_facets), V),
        ],
        [
            dolfinx.fem.dirichletbc(
                0.0, dolfinx.fem.locate_dofs_topological(V, dim, left_facets), V
            ),
            dolfinx.fem.dirichletbc(0.0, dolfinx.fem.locate_dofs_topological(V, dim, tb_facets), V),
        ],
    ]
    n = ufl.FacetNormal(mesh)
    v = ufl.TestFunction(V)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags, subdomain_id=right_value)
    A = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(a))
    A.assemble()
    for bc in bcs_u[0]:
        A.zeroRowsLocal(bc._cpp_object.dof_indices()[0], 1.0)
    bs = []
    for i in range(mesh.geometry.dim):
        b = dolfinx.fem.petsc.assemble_vector(dolfinx.fem.form(Ls[i]))
        dolfinx.fem.petsc.assemble_vector(b, dolfinx.fem.form(p_value * v.dx(i) * n[i] * ds))
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.petsc.set_bc(b, bcs_u[i])
        b.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        bs.append(b)

    # Compare matrices
    root = 0
    scipy_A = gather_PETScMatrix(A, mesh.comm, root=root)
    scipy_A_oasis = gather_PETScMatrix(A_oasis, mesh.comm, root=root)
    if mesh.comm == root:
        diff = np.abs(scipy_A, scipy_A_oasis)
        assert diff.max() < 1e-14

    # Compare vectors
    converged_u = solver.velocity_tentative_solve()
    for i in range(mesh.geometry.dim):
        assert converged_u[1][i]
        assert np.allclose(solver._rhs1[i].vector.array, bs[i].array)

    # solver.pressure_assemble(dt)
    # converged_p = solver.pressure_solve()
    # assert converged_p
    # # converged_update = solver.velocity_update(dt)

    [b.destroy() for b in bs]
