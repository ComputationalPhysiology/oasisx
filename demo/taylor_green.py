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
import logging
import oasisx
logger = logging.getLogger("Oasisx")

N = 50
dt = 1.e-2
nu = 0.01
T_end = 1
T_start = 0

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
        return - np.cos(np.pi * x[0]) * np.sin(np.pi * x[1]) * np.exp(-2.0 * self.nu * np.pi**2 * float(self.t))

    def eval_y(self, x):
        return np.cos(np.pi * x[1]) * np.sin(np.pi * x[0]) * np.exp(-2.0 * self.nu * np.pi**2 * float(self.t))


assert (T_start < T_end)

num_steps = int((T_end-T_start)//dt)

u_time = dolfinx.fem.Constant(mesh, np.float64(T_start))
p_time = dolfinx.fem.Constant(mesh, np.float64(T_start-dt/2.))
u_ex = U(t=u_time, nu=nu)

bcx = oasisx.DirichletBC(u_ex.eval_x, oasisx.LocatorMethod.TOPOLOGICAL, (facet_tags, value))
bcy = oasisx.DirichletBC(u_ex.eval_y, oasisx.LocatorMethod.TOPOLOGICAL, (facet_tags, value))

bcs_u = [[bcx], [bcy]]
bcs_p = []
# Create fractional step solver
solver = oasisx.FractionalStep_AB_CN(
    mesh, el_u, el_p, bcs_u=bcs_u, bcs_p=bcs_p,
    solver_options=solver_options, options=options, body_force=f)

# Set initial conditions for velocity
u_time.value = T_start-dt
solver._u2[0].interpolate(u_ex.eval_x)
solver._u2[1].interpolate(u_ex.eval_y)
u_time.value = T_start
solver._u1[0].interpolate(u_ex.eval_x)
solver._u1[1].interpolate(u_ex.eval_y)

# Set initial conditions for pressure
x = ufl.SpatialCoordinate(mesh)
p_ex = -1/4 * (ufl.cos(2*ufl.pi*x[0])+ufl.cos(2*ufl.pi*x[1]))*ufl.exp(-4*ufl.pi**2*nu*p_time)
p_expr = dolfinx.fem.Expression(p_ex, solver._Q.element.interpolation_points())
solver._p.interpolate(p_expr)
vtxu = dolfinx.io.VTXWriter(mesh.comm, "u.bp", [solver.u])
vtxp = dolfinx.io.VTXWriter(mesh.comm, "p.bp", [solver._p])

man_u = ufl.as_vector((
    - ufl.sin(ufl.pi*x[1])*ufl.cos(ufl.pi*x[0]),
    ufl.sin(ufl.pi*x[0])*ufl.cos(ufl.pi*x[1]))
)*ufl.exp(-2*ufl.pi**2*nu*u_time)
diff_u = solver.u-man_u
L2_u = dolfinx.fem.form(ufl.inner(diff_u, diff_u) * ufl.dx)
diff_p = solver._p - p_ex
L2_p = dolfinx.fem.form(ufl.inner(diff_p, diff_p) * ufl.dx)
i = 0

error_space_time = np.empty((2, num_steps), dtype=np.float64)
u_time.value = T_start
for i in range(num_steps):
    u_time.value += dt
    p_time.value += dt

    solver.solve(dt, nu, max_iter=1)
    L2_u_loc = dolfinx.fem.assemble_scalar(L2_u)
    error_u = mesh.comm.allreduce(L2_u_loc, op=MPI.SUM)
    L2_p_loc = dolfinx.fem.assemble_scalar(L2_p)
    error_p = mesh.comm.allreduce(L2_p_loc, op=MPI.SUM)
    logger.debug(f"{u_ex.t=}, {error_u=}")
    logger.debug(f"{float(p_time.value)}, {error_p=}")
    vtxp.write(p_time.value)
    vtxu.write(u_time.value)
    error_space_time[:, i] = [error_u, error_p]
vtxu.close()
vtxp.close()
space_time_u_L2 = np.sqrt(dt*np.sum(error_space_time[0, :]))
space_time_p_L2 = np.sqrt(dt*np.sum(error_space_time[1, :]))
logger.setLevel(logging.INFO)
logger.info(f"{space_time_u_L2=} {space_time_p_L2}")
