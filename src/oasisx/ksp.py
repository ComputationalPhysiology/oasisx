# Copyright (C) 2022 Jørgen Schartum Dokken
#
# This file is part of Oasisx
# SPDX-License-Identifier:    MIT

import typing

from mpi4py import MPI
from petsc4py import PETSc as _PETSc

import dolfinx.fem as _fem
import numpy as np


class KSPSolver:
    _prefix: str
    _ksp: _PETSc.KSP  # type: ignore
    __slots__ = tuple(__annotations__)

    def __init__(
        self,
        comm: MPI.Comm,
        petsc_options: typing.Optional[dict] = None,
        prefix="oasis_solver",
    ):
        """Lightweight wrapper around the PETSc KSP solver
        Args:
            comm: MPI communicator used in PETSc Solver
            petsc_options: Options that are passed to the linear
                algebra backend PETSc. For available choices for the
                'petsc_options' kwarg, see the `PETSc documentation
                <https://petsc4py.readthedocs.io/en/stable/manual/ksp/>`_.
        """
        petsc_options = {} if petsc_options is None else petsc_options
        self._ksp = _PETSc.KSP().create(comm)  # type: ignore
        self._prefix = prefix
        self.updateOptions(petsc_options)

    def updateOptions(self, options: dict):
        """
        Update PETSc options.

        ..note::
            that :func:`setOptions` has to be called after this operation
        """
        self._ksp.setOptionsPrefix(self._prefix)
        opts = _PETSc.Options()  # type: ignore
        opts.prefixPush(self._prefix)
        for k, v in options.items():
            opts[k] = v
        opts.prefixPop()
        self._ksp.setFromOptions()
        for opt in opts.getAll().keys():
            del opts[opt]

    def setOptions(self, op: typing.Union[_PETSc.Mat, _PETSc.Vec]):  # type: ignore
        prefix = self._ksp.getOptionsPrefix()
        assert prefix is not None
        op.setOptionsPrefix(prefix)
        op.setFromOptions()

    def setOperators(
        self,
        A: _PETSc.Mat,  # type: ignore
        P: typing.Optional[_PETSc.Mat] = None,  # type: ignore
    ):
        if P is None:
            self._ksp.setOperators(A)
        else:
            self._ksp.setOperators(A, P)

    def solve(
        self,
        b: _PETSc.Vec,  # type: ignore
        x: _fem.Function,
    ) -> np.int32:
        self._ksp.solve(b, x.vector)
        x.x.scatter_forward()
        return np.int32(self._ksp.getConvergedReason())

    def __del__(self):
        """
        Delete PETSc options manually due to https://gitlab.com/petsc/petsc/-/issues/1201
        """
        try:
            opts = _PETSc.Options()  # type: ignore
            all_opts = opts.getAll()
            for key in all_opts.keys():
                if f"Oasis_solve_{id(self)}" in key:
                    opts.delValue(key)
        except TypeError:
            pass
