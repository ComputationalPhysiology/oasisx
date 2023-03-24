# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
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


import ufl
import dolfinx
import numpy as np
from mpi4py import MPI

import oasisx

N = 25
mesh = dolfinx.mesh.create_rectangle(
    MPI.COMM_WORLD, [[-1, -1], [1, 1]], [N, N], cell_type=dolfinx.mesh.CellType.triangle)
dim = mesh.topology.dim - 1
el_u = ("Lagrange", 2)
el_p = ("Lagrange", 1)

solver_options = {"tentative": {"ksp_type": "preonly", "pc_type": "lu"},
                  "pressure": {"ksp_type": "preonly", "pc_type": "lu"},
                  "scalar": {"ksp_type": "preonly", "pc_type": "lu"}}
options = {"low_memory_version": True}
f = None


# Locate facets for boundary conditions and create  meshtags
mesh.topology.create_connectivity(dim, dim+1)
facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
value = 3
values = np.full_like(facets, value, dtype=np.int32)
sort = np.argsort(facets)
facet_tags = dolfinx.mesh.meshtags(mesh, dim, facets[sort], values[sort])

# Create boundary conditions


class U():
    def __init__(self, t, nu):
        self.t = t
        self.nu = nu

    def eval_x(self, x):
        return - np.cos(np.pi * x[0]) * np.sin(np.pi * x[1]) * np.exp(-2.0 * self.nu * np.pi**2 * self.t)

    def eval_y(self, x):
        return np.cos(np.pi * x[1]) * np.sin(np.pi * x[0]) * np.exp(-2.0 * self.nu * np.pi**2 * self.t)


dt = 1.e-2
nu = 0.01
u_ex = U(t=0, nu=nu)

bcx = oasisx.DirichletBC(u_ex.eval_x, oasisx.LocatorMethod.TOPOLOGICAL, (facet_tags, value))
bcy = oasisx.DirichletBC(u_ex.eval_y, oasisx.LocatorMethod.TOPOLOGICAL, (facet_tags, value))

bcs_u = [[bcx], [bcy]]
bcs_p = []
# Create fractional step solver
solver = oasisx.FractionalStep_AB_CN(
    mesh, el_u, el_p, bcs_u=bcs_u, bcs_p=bcs_p,
    solver_options=solver_options, options=options, body_force=f)

# Set initial conditions for velocity
u_ex.t = -dt
solver._u2[0].interpolate(u_ex.eval_x)
solver._u2[1].interpolate(u_ex.eval_y)
u_ex.t = 0
solver._u1[0].interpolate(u_ex.eval_x)
solver._u1[1].interpolate(u_ex.eval_y)
# Set initial conditions for pressure
x = ufl.SpatialCoordinate(mesh)
p_time = dolfinx.fem.Constant(mesh, -dt/2.)
p_ex = -1/4 * (ufl.cos(2*ufl.pi*x[0])+ufl.cos(2*ufl.pi*x[1]))*ufl.exp(-4*ufl.pi**2*nu*p_time)
p_expr = dolfinx.fem.Expression(p_ex, solver._Q.element.interpolation_points())
solver._p.interpolate(p_expr)
vtxu = dolfinx.io.VTXWriter(mesh.comm, "u.bp", [solver.u])
vtxp = dolfinx.io.VTXWriter(mesh.comm, "p.bp", [solver._p])

u_time = dolfinx.fem.Constant(mesh, 0.)
man_u = ufl.as_vector((
    - ufl.sin(ufl.pi*x[1])*ufl.cos(ufl.pi*x[0]),
    ufl.sin(ufl.pi*x[0])*ufl.cos(ufl.pi*x[1]))
)*ufl.exp(-2*ufl.pi**2*nu*u_time)
diff_u = solver.u-man_u
L2_u = dolfinx.fem.form(ufl.inner(diff_u, diff_u) * ufl.dx)
diff_p = solver._p - p_ex
L2_p = dolfinx.fem.form(ufl.inner(diff_p, diff_p) * ufl.dx)
i = 0


u_ex.t = 0

while u_ex.t < 0.5:
    u_ex.t += dt
    u_time.value = u_ex.t
    p_time.value += dt

    solver.solve(dt, nu, max_iter=1)
    L2_u_loc = dolfinx.fem.assemble_scalar(L2_u)
    error_u = np.sqrt(mesh.comm.allreduce(L2_u_loc, op=MPI.SUM))
    L2_p_loc = dolfinx.fem.assemble_scalar(L2_p)
    error_p = np.sqrt(mesh.comm.allreduce(L2_p_loc, op=MPI.SUM))
    print(f"{u_ex.t=}, {error_u=}")
    print(f"{float(p_time.value)}, {error_p=}")
    vtxp.write(p_time.value)
    vtxu.write(u_time.value)
vtxu.close()
vtxp.close()
