# Copyright (C) 2022 Jørgen Schartum Dokken
#
# This file is part of Oasisx
# SPDX-License-Identifier:    MIT

from petsc4py import PETSc as _PETSc
import dolfinx.fem as _fem
import typing
from mpi4py import MPI


class KSPSolver:
    _ksp: _PETSc.KSP  # Krylov subspace solver
    __slots__ = tuple(__annotations__)

    def __init__(self, comm: MPI.Comm, petsc_options: dict = None):
        """Lightweight wrapper around the PETSc KSP solver
        Args:
            comm: MPI communicator used in PETSc Solver
            petsc_options: Options that are passed to the linear
                algebra backend PETSc. For available choices for the
                'petsc_options' kwarg, see the `PETSc documentation
                <https://petsc4py.readthedocs.io/en/stable/manual/ksp/>`_.
        """
        petsc_options = {} if petsc_options is None else petsc_options
        self._ksp = _PETSc.KSP().create(comm)
        prefix = f"Oasis_solve_{id(self)}"
        self._ksp.setOptionsPrefix(prefix)
        opts = _PETSc.Options()
        opts.prefixPush(prefix)
        for k, v in petsc_options.items():
            opts[k] = v
        opts.prefixPop()
        self._ksp.setFromOptions()

    def setOptions(self, op: typing.Union[_PETSc.Mat, _PETSc.Vec]):
        prefix = self._ksp.getOptionsPrefix()
        assert prefix is not None
        op.setOptionsPrefix(prefix)
        op.setFromOptions()

    def setOperators(self, A: _PETSc.Mat, P: typing.Optional[_PETSc.Mat] = None):
        if P is None:
            self._ksp.setOperators(A)
        else:
            self._ksp.setOperators(A, P)

    def solve(self, b: _PETSc.Vec, x: _fem.Function):
        self._ksp.solve(b, x.vector)
        x.x.scatter_forward()

    def __del__(self):
        """
        Delete PETSc options manually due to https://gitlab.com/petsc/petsc/-/issues/1201
        """
        opts = _PETSc.Options()
        all_opts = opts.getAll()
        for key in all_opts.keys():
            if f"Oasis_solve_{id(self)}" in key:
                opts.delValue(key)
