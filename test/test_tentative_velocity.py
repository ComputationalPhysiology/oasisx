# Copyright (C) 2022 Jørgen Schartum Dokken
#
# This file is part of Oasisx
# SPDX-License-Identifier:    MIT


from oasisx import FractionalStep_AB_CN
import dolfinx
from mpi4py import MPI
import numpy as np


def test_tentative():
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)
    el_u = ("Lagrange", 2)
    el_p = ("Lagrange", 1)

    solver_options = {"tentative": {"ksp_type": "preonly", "pc_type": "lu"}}
    options = {"low_memory_version": True}
    body_force = np.array([1., 2., 3], dtype=np.float64)

    solver = FractionalStep_AB_CN(
        mesh, el_u, el_p, solver_options=solver_options, options=options, body_force=body_force)
    solver.tenative_velocity(0.1, 1.)
