# Copyright (C) 2022 Jørgen Schartum Dokken
#
# This file is part of Oasisx
# SPDX-License-Identifier:    MIT
#
# Tests for DirichletBC wrapper

from mpi4py import MPI

import dolfinx
import dolfinx.fem.petsc
import numpy as np
import pytest
import ufl

from oasisx import DirichletBC, LocatorMethod, PressureBC


@pytest.mark.parametrize("P", np.arange(1, 5))
def test_function_geometrical(P):
    """
    Test assignement of a time dependent function to DirichletBC using geometrical mode
    """
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)

    def locator(x):
        return np.isclose(x[0], 1)

    class TimeDependentBC:
        def __init__(self, t: float):
            self.t = t

        def eval(self, x):
            return np.sin(x[0]) + x[1] * self.t

    condition_0 = TimeDependentBC(0.1)

    bc = DirichletBC(condition_0.eval, LocatorMethod.GEOMETRICAL, locator)

    V = dolfinx.fem.functionspace(mesh, ("Lagrange", int(P)))
    bc.create_bc(V)

    for t in [0.1, 0.2, 0.3]:
        u = dolfinx.fem.Function(V)
        u.interpolate(lambda x: np.sin(x[0]) + x[1] * t)
        bc_dx = dolfinx.fem.dirichletbc(u, dolfinx.fem.locate_dofs_geometrical(V, locator))

        u_bcx = dolfinx.fem.Function(V)
        dolfinx.fem.petsc.set_bc(u_bcx.vector, [bc_dx])

        u_bc = dolfinx.fem.Function(V)
        condition_0.t = t
        bc.update_bc()
        bc.apply(u_bc.vector)
        assert np.allclose(u_bcx.x.array, u_bc.x.array)


@pytest.mark.parametrize("P", np.arange(1, 5))
@pytest.mark.parametrize("dim", np.arange(0, 3))
def test_function_topological(P, dim):
    """
    Test assignement of a time dependent function to DirichletBC using topological mode
    """
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)

    def locator(x):
        return np.isclose(x[0], 1)

    class TimeDependentBC:
        def __init__(self, t: float):
            self.t = t

        def eval(self, x):
            return np.sin(x[0]) + x[1] * self.t

    condition_0 = TimeDependentBC(0.1)
    entities = dolfinx.mesh.locate_entities(mesh, dim, locator)
    value = np.int32(3)
    et = dolfinx.mesh.meshtags(mesh, dim, entities, np.full(len(entities), value, dtype=np.int32))
    bc = DirichletBC(condition_0.eval, LocatorMethod.TOPOLOGICAL, (et, value))

    V = dolfinx.fem.functionspace(mesh, ("Lagrange", int(P)))
    bc.create_bc(V)

    for t in [0.1, 0.2, 0.3]:
        u = dolfinx.fem.Function(V)
        u.interpolate(lambda x: np.sin(x[0]) + x[1] * t)
        bc_dx = dolfinx.fem.dirichletbc(u, dolfinx.fem.locate_dofs_topological(V, dim, entities))

        u_bcx = dolfinx.fem.Function(V)
        dolfinx.fem.petsc.set_bc(u_bcx.vector, [bc_dx])

        u_bc = dolfinx.fem.Function(V)
        condition_0.t = t
        bc.update_bc()
        bc.apply(u_bc.vector)
        assert np.allclose(u_bcx.x.array, u_bc.x.array)


@pytest.mark.parametrize("P", np.arange(1, 5))
def test_constant_geometrical(P):
    """
    Test assignement of a time dependent constant to DirichletBC using geometrical mode
    """
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)

    def locator(x):
        return np.isclose(x[0], 1)

    time = dolfinx.fem.Constant(mesh, 1.0)

    bc = DirichletBC(time, LocatorMethod.GEOMETRICAL, locator)

    V = dolfinx.fem.functionspace(mesh, ("Lagrange", int(P)))
    bc.create_bc(V)

    for t in [0.1, 0.2, 0.3]:
        time.value += t
        bc_dx = dolfinx.fem.dirichletbc(time, dolfinx.fem.locate_dofs_geometrical(V, locator), V)
        u_bcx = dolfinx.fem.Function(V)
        dolfinx.fem.petsc.set_bc(u_bcx.vector, [bc_dx])

        u_bc = dolfinx.fem.Function(V)
        bc.apply(u_bc.vector)
        assert np.allclose(u_bcx.x.array, u_bc.x.array)


@pytest.mark.parametrize("P", np.arange(1, 5))
@pytest.mark.parametrize("dim", np.arange(0, 3))
def test_constant_topological(P, dim):
    """
    Test assignement of a time dependent constant to DirichletBC using geometrical mode
    """

    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)

    def locator(x):
        return np.isclose(x[0], 1)

    time = dolfinx.fem.Constant(mesh, 1.0)

    entities = dolfinx.mesh.locate_entities(mesh, dim, locator)
    value = np.int32(3)
    et = dolfinx.mesh.meshtags(mesh, dim, entities, np.full(len(entities), value, dtype=np.int32))
    bc = DirichletBC(time, LocatorMethod.TOPOLOGICAL, (et, value))

    V = dolfinx.fem.functionspace(mesh, ("Lagrange", int(P)))
    bc.create_bc(V)

    for t in [0.1, 0.2, 0.3]:
        time.value += t
        u = dolfinx.fem.Function(V)
        u.interpolate(lambda x: np.sin(x[0]) + x[1] * t)
        bc_dx = dolfinx.fem.dirichletbc(
            time, dolfinx.fem.locate_dofs_topological(V, dim, entities), V
        )

        u_bcx = dolfinx.fem.Function(V)
        dolfinx.fem.petsc.set_bc(u_bcx.vector, [bc_dx])

        u_bc = dolfinx.fem.Function(V)
        bc.apply(u_bc.vector)
        assert np.allclose(u_bcx.x.array, u_bc.x.array)


@pytest.mark.parametrize("P", np.arange(2, 4))
def test_pressure_condition(P):
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)

    def locator(x):
        return np.isclose(x[0], 1)

    class TimeDependentBC:
        def __init__(self, t: float):
            self.t = t

        def eval(self, x):
            return np.sin(x[0]) + x[1] * self.t

    condition_0 = TimeDependentBC(0.1)

    entities = dolfinx.mesh.locate_entities(mesh, mesh.topology.dim - 1, locator)
    value = np.int32(3)
    et = dolfinx.mesh.meshtags(
        mesh,
        mesh.topology.dim - 1,
        entities,
        np.full(len(entities), value, dtype=np.int32),
    )
    bc = PressureBC(condition_0.eval, (et, value))
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", int(P)))
    Q = dolfinx.fem.functionspace(mesh, ("Lagrange", int(P - 1)))
    bc.create_bcs(V, Q)
    p = dolfinx.fem.Function(Q)
    p.interpolate(condition_0.eval)
    v = ufl.TestFunction(V)
    n = ufl.FacetNormal(mesh)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=et, subdomain_id=value)
    for i, ni in enumerate(n):
        rhs = dolfinx.fem.form(p * n[i] * v.dx(i) * ds)
        b_form = dolfinx.fem.Function(V)
        dolfinx.fem.petsc.assemble_vector(b_form.vector, rhs)
        b_bc = dolfinx.fem.Function(V)
        dolfinx.fem.petsc.assemble_vector(b_bc.vector, dolfinx.fem.form(bc.rhs(i)))
        assert np.allclose(b_form.x.array, b_bc.x.array)

    dofs = dolfinx.fem.locate_dofs_topological(Q, mesh.topology.dim - 1, et.find(value))
    bc_ex = dolfinx.fem.dirichletbc(0.0, dofs, Q)
    r = dolfinx.fem.Function(Q)
    r.x.array[:] = 10
    dolfinx.fem.petsc.set_bc(r.vector, [bc_ex])

    s = dolfinx.fem.Function(Q)
    s.x.array[:] = 10
    dolfinx.fem.petsc.set_bc(s.vector, [bc.bc])

    assert np.allclose(r.x.array, s.x.array)
