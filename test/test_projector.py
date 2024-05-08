# Copyright (C) 2022 Jørgen Schartum Dokken
#
# This file is part of Oasisx
# SPDX-License-Identifier:    MIT

from mpi4py import MPI

import basix.ufl
import dolfinx
import numpy as np
import ufl

from oasisx import Projector


def test_projector():
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 2))

    # Interpolate initial solutiom
    u = dolfinx.fem.Function(V)
    u.interpolate(lambda x: x[0] * x[0] + 3 * x[1] + 2 * x[1] * x[1])

    # Create gradient projector
    el = basix.ufl.element("DG", mesh.topology.cell_name(), 1, shape=(mesh.geometry.dim,))
    W = dolfinx.fem.functionspace(mesh, el)
    petsc_options = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    gradient_projector = Projector(ufl.grad(u), W, [], petsc_options=petsc_options)
    gradient_projector.solve()

    # assemle L2 error
    x = ufl.SpatialCoordinate(mesh)
    u_ex = ufl.as_vector((2 * x[0], 3 + 4 * x[1]))
    ph = gradient_projector.x
    error_form = dolfinx.fem.form(ufl.inner(u_ex - ph, u_ex - ph) * ufl.dx)
    L2_squared = dolfinx.fem.assemble_scalar(error_form)
    u.interpolate(lambda x: x[0] + 2 * x[1] * x[1])
    assert np.isclose(np.sqrt(L2_squared), 0.0, atol=1e-12)

    gradient_projector.assemble_rhs()
    gradient_projector.solve(assemble_rhs=False)

    u_ex_new = ufl.as_vector((1, 4 * x[1]))
    new_error = dolfinx.fem.form(ufl.inner(u_ex_new - ph, u_ex_new - ph) * ufl.dx)
    L2_squared_new = dolfinx.fem.assemble_scalar(new_error)
    assert np.isclose(np.sqrt(L2_squared_new), 0.0, atol=1e-12)
