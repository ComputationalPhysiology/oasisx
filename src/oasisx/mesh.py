# Copyright (C) 2022 Jørgen Schartum Dokken
#
# This file is part of Oasisx
# SPDX-License-Identifier:    MIT


__all__ = ["import_mesh"]

from mpi4py import MPI

import dolfinx.mesh as _mesh


def import_mesh(filename: str) -> _mesh.Mesh:
    print(filename)
    return _mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)
