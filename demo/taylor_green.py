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


import argparse
import logging
from typing import List

from mpi4py import MPI

import dolfinx
import numpy as np
import numpy.typing as npt
import ufl

import oasisx


class U:
    def __init__(self, t, nu):
        self.t = t
        self.nu = nu

    def eval_x(self, x: npt.NDArray[np.float64]) -> npt.NDArray[dolfinx.default_scalar_type]:
        return (
            -np.cos(np.pi * x[0])
            * np.sin(np.pi * x[1])
            * np.exp(-2.0 * self.nu * np.pi**2 * float(self.t))
        )

    def eval_y(self, x: npt.NDArray[np.float64]) -> npt.NDArray[dolfinx.default_scalar_type]:
        return (
            np.cos(np.pi * x[1])
            * np.sin(np.pi * x[0])
            * np.exp(-2.0 * self.nu * np.pi**2 * float(self.t))
        )


desc = "Taylor-Green convergence demo"
parser = argparse.ArgumentParser(
    description=desc, formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    "-N",
    "--refinement",
    type=int,
    dest="Ns",
    action="append",
    help="The number of elements in x and y direction",
    required=True,
)
parser.add_argument(
    "-T0",
    "--T-start",
    dest="T_start",
    type=float,
    help="Start time of simulation",
    default=0,
)
parser.add_argument(
    "-T1", "--T-end", dest="T_end", type=float, help="End time of simulation", default=1
)
parser.add_argument("-dt", dest="dt", type=float, help="Time step", default=0.1)
parser.add_argument("-nu", dest="nu", type=float, help="Kinematic viscosity", default=0.01)
parser.add_argument("-u", dest="u_deg", type=int, help="Degree of velocity space", default=2)
parser.add_argument("-p", dest="p_deg", type=int, help="Degree of pressure space", default=1)
parser.add_argument(
    "-lm",
    "--low-memory",
    dest="lm",
    action="store_true",
    default=False,
    help="Use low memory version of Oasisx",
)
parser.add_argument(
    "-r",
    "--rotational",
    dest="rot",
    action="store_true",
    default=False,
    help="Use rotational formulation of pressure update",
)
inputs = parser.parse_args()
# FIXME: add loglevel to input parser
logger = logging.getLogger("Oasisx")

dt = inputs.dt
nu = inputs.nu
assert inputs.T_start < inputs.T_end
T_end = inputs.T_end
T_start = inputs.T_start
num_steps = int((T_end - T_start) // dt)

assert inputs.u_deg > inputs.p_deg
el_u = ("Lagrange", inputs.u_deg)
el_p = ("Lagrange", inputs.p_deg)
f = None
options = {"low_memory_version": inputs.lm}

solver_options = {
    "tentative": {"ksp_type": "preonly", "pc_type": "lu"},
    "pressure": {"ksp_type": "preonly", "pc_type": "lu"},
    "scalar": {"ksp_type": "preonly", "pc_type": "lu"},
}

space_errors = np.zeros((2, len(inputs.Ns)), dtype=np.float64)
hs = np.zeros(len(inputs.Ns), dtype=np.float64)
for n, N in enumerate(inputs.Ns):
    mesh = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD,
        [[-1, -1], [1, 1]],
        [N, N],
        cell_type=dolfinx.mesh.CellType.triangle,
    )
    dim = mesh.topology.dim - 1

    # Locate facets for boundary conditions and create  meshtags
    mesh.topology.create_connectivity(dim, dim + 1)
    facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    value = np.int32(3)
    values = np.full_like(facets, value, dtype=np.int32)
    sort = np.argsort(facets)
    facet_tags = dolfinx.mesh.meshtags(mesh, dim, facets[sort], values[sort])

    u_time = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(T_start))
    p_time = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(T_start - dt / 2.0))
    u_ex = U(t=u_time, nu=nu)

    bcx = oasisx.DirichletBC(u_ex.eval_x, oasisx.LocatorMethod.TOPOLOGICAL, (facet_tags, value))
    bcy = oasisx.DirichletBC(u_ex.eval_y, oasisx.LocatorMethod.TOPOLOGICAL, (facet_tags, value))

    bcs_u = [[bcx], [bcy]]
    bcs_p: List[oasisx.PressureBC] = []
    # Create fractional step solver
    solver = oasisx.FractionalStep_AB_CN(
        mesh,
        el_u,
        el_p,
        bcs_u=bcs_u,
        bcs_p=bcs_p,
        rotational=inputs.rot,
        solver_options=solver_options,
        options=options,
        body_force=f,
    )

    # Set initial conditions for velocity
    u_time.value = T_start - dt
    solver._u2[0].interpolate(u_ex.eval_x)
    solver._u2[1].interpolate(u_ex.eval_y)
    u_time.value = T_start
    solver._u1[0].interpolate(u_ex.eval_x)
    solver._u1[1].interpolate(u_ex.eval_y)

    # Set initial conditions for pressure
    x = ufl.SpatialCoordinate(mesh)
    man_p = (
        -0.25
        * (ufl.cos(2 * ufl.pi * x[0]) + ufl.cos(2 * ufl.pi * x[1]))
        * ufl.exp(-4 * ufl.pi**2 * nu * p_time)
    )
    p_expr = dolfinx.fem.Expression(man_p, solver._Q.element.interpolation_points())
    solver._p.interpolate(p_expr)
    vtxu = dolfinx.io.VTXWriter(mesh.comm, "u.bp", [solver.u], engine="BP4")
    vtxp = dolfinx.io.VTXWriter(mesh.comm, "p.bp", [solver._p], engine="BP4")

    man_u = ufl.as_vector(
        (
            -ufl.sin(ufl.pi * x[1]) * ufl.cos(ufl.pi * x[0]),
            ufl.sin(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1]),
        )
    ) * ufl.exp(-2 * ufl.pi**2 * nu * u_time)
    diff_u = solver.u - man_u
    L2_u = dolfinx.fem.form(ufl.inner(diff_u, diff_u) * ufl.dx)
    diff_p = solver._p - man_p
    L2_p = dolfinx.fem.form(ufl.inner(diff_p, diff_p) * ufl.dx)

    error_space_time = np.zeros((2, num_steps), dtype=np.float64)
    u_time.value = T_start
    for i in range(num_steps):
        u_time.value += dt
        p_time.value += dt

        solver.solve(dt, nu, max_iter=1)
        L2_u_loc = dolfinx.fem.assemble_scalar(L2_u)
        error_u = mesh.comm.allreduce(L2_u_loc, op=MPI.SUM)
        L2_p_loc = dolfinx.fem.assemble_scalar(L2_p)
        error_p = mesh.comm.allreduce(L2_p_loc, op=MPI.SUM)
        logger.debug(f"{float(u_time.value)}, {error_u=}")
        logger.debug(f"{float(p_time.value)}, {error_p=}")
        logger.debug("*" * 10)
        vtxp.write(p_time.value)
        vtxu.write(u_time.value)
        error_space_time[:, i] = [error_u, error_p]

    vtxu.close()
    vtxp.close()
    mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim)
    hmax_loc = np.max(
        mesh.h(
            mesh.topology.dim,
            np.arange(mesh.topology.index_map(mesh.topology.dim).size_local, dtype=np.int32),
        )
    )
    hmax = mesh.comm.allreduce(hmax_loc, op=MPI.MAX)
    space_time_u_L2 = np.sqrt(dt * np.sum(error_space_time[0, :]))
    space_time_p_L2 = np.sqrt(dt * np.sum(error_space_time[1, :]))

    logger.setLevel(logging.INFO)
    logger.info(f"{hmax=} {space_time_u_L2=} {space_time_p_L2=}")
    hs[n] = hmax
    space_errors[:, n] = [space_time_u_L2, space_time_p_L2]

order = np.argsort(hs)[::-1]
hs = hs[order]

space_errors[0, :] = space_errors[0, order]
space_errors[1, :] = space_errors[1, order]
rate_u = np.log(space_errors[0, 1:] / space_errors[0, :-1]) / np.log(hs[1:] / hs[:-1])
rate_p = np.log(space_errors[1, 1:] / space_errors[1, :-1]) / np.log(hs[1:] / hs[:-1])
logger.info(f"Convergence rates u: {rate_u}")
logger.info(f"Convergence rates p: {rate_p}")
