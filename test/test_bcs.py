# Copyright (C) 2022 Jørgen Schartum Dokken
#
# This file is part of Oasisx
# SPDX-License-Identifier:    MIT


import dolfinx
import numpy as np
import pytest
from mpi4py import MPI
from oasisx import DirichletBC, LocatorMethod


def test_geometrical():
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)

    def locator(x):
        return np.isclose(x[0], 0)

    def value(x):
        return np.sin(x[0])

    bc = DirichletBC(value, geometrical=locator)

    V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 1))
    bc.create_bc(V)
    from IPython import embed
    embed()
