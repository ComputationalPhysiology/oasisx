# Copyright (C) 2022 Jørgen Schartum Dokken
#
# This file is part of Oasisx
# SPDX-License-Identifier:    MIT


import argparse

from .fracstep import FractionalStep_AB_CN
from .mesh import import_mesh

desc = (
    "Welcome to the Oasisx Python Package. To run the code, add the"
    + " following command-line arguments"
)

parser = argparse.ArgumentParser(
    description=desc, formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    "--mesh-path", required=True, type=str, dest="mesh_file", help="Path to mesh file"
)


def main(args=None):
    xargs = parser.parse_args(args)
    mesh = import_mesh(xargs.mesh_file)
    problem = FractionalStep_AB_CN(mesh, ("Lagrange", 2), ("Lagrange", 1))
    problem.assemble_first(0.1, 0.2)
