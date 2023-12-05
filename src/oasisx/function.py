# Copyright (C) 2022 Jørgen Schartum Dokken
#
# This file is part of Oasisx
# SPDX-License-Identifier:    MIT

from typing import List, Optional

import dolfinx.fem
import ufl
from petsc4py import PETSc as _petsc


class Projector():
    """
    Projector for a given function.
    Solves Ax=b, where

    .. highlight:: python
    .. code-block:: python

        u, v = ufl.TrialFunction(Space), ufl.TestFunction(space)
        dx = ufl.Measure("dx", metadata=metadata)
        A = inner(u, v) * dx
        b = inner(function, v) * dx

    Args:
        function: UFL expression of function to project
        space: Space to project function into
        bcs: List of BCS to apply to projection
        petsc_options: Options to pass to PETSc
        jit_options: Options to pass to just in time compiler
        form_compiler_options: Options to pass to the form compiler
        metadata: Data to pass to the integration measure
    """
    # The mass matrix
    _A: _petsc.Mat  # type: ignore
    # The rhs vector
    _b: _petsc.Vec  # type: ignore
    _lhs: dolfinx.fem.forms.Form  # The compiled form for the mass matrix
    _rhs: dolfinx.fem.forms.Form  # The compiled form form the rhs vector
    # The PETSc solver
    _ksp: _petsc.KSP  # type: ignore
    _x: dolfinx.fem.Function  # The solution vector
    _bcs: List[dolfinx.fem.DirichletBC]
    __slots__ = tuple(__annotations__)

    def __init__(self, function: ufl.core.expr.Expr,
                 space: dolfinx.fem.FunctionSpace,
                 bcs: List[dolfinx.fem.DirichletBC],
                 petsc_options: Optional[dict] = None,
                 jit_options: Optional[dict] = None,
                 form_compiler_options: Optional[dict] = None,
                 metadata: Optional[dict] = None):
        petsc_options = {} if petsc_options is None else petsc_options
        jit_options = {} if jit_options is None else jit_options
        form_compiler_options = {} if form_compiler_options is None else form_compiler_options

        # Assemble projection matrix once
        u = ufl.TrialFunction(space)
        v = ufl.TestFunction(space)
        a = ufl.inner(u, v) * ufl.dx(metadata=metadata)
        self._lhs = dolfinx.fem.form(a, jit_options=jit_options,
                                     form_compiler_options=form_compiler_options)
        self._A = dolfinx.fem.petsc.assemble_matrix(self._lhs, bcs=bcs)
        self._A.assemble()

        # Compile RHS form and create vector
        L = ufl.inner(function, v) * ufl.dx(metadata=metadata)
        self._rhs = dolfinx.fem.form(L, jit_options=jit_options,
                                     form_compiler_options=form_compiler_options)
        self._x = dolfinx.fem.Function(space)
        self._b = dolfinx.fem.Function(space)
        self._bcs = bcs

        # Create Krylov Subspace solver
        self._ksp = _petsc.KSP().create(space.mesh.comm)  # type: ignore
        self._ksp.setOperators(self._A)

        # Set PETSc options
        prefix = f"oasis_projector_{id(self)}"
        opts = _petsc.Options()  # type: ignore
        opts.prefixPush(prefix)
        for k, v in petsc_options.items():
            opts[k] = v
        opts.prefixPop()
        self._ksp.setFromOptions()

        # Set matrix and vector PETSc options
        self._A.setOptionsPrefix(prefix)
        self._A.setFromOptions()
        self._b.vector.setOptionsPrefix(prefix)
        self._b.vector.setFromOptions()

    def assemble_rhs(self):
        """
        Update RHS by re-assembling
        """
        self._b.x.array[:] = 0.
        dolfinx.fem.petsc.assemble_vector(self._b.vector, self._rhs)
        dolfinx.fem.petsc.apply_lifting(
            self._b.vector, [self._lhs], bcs=[self._bcs])
        self._b.x.scatter_reverse(dolfinx.cpp.la.InsertMode.add)
        dolfinx.fem.petsc.set_bc(self._b.vector, self._bcs)
        self._b.x.scatter_forward()

    def solve(self, assemble_rhs: bool = True):
        """
        Compute projection using PETSc a KSP solver

        Args:
            assemble_rhs: Re-assemble RHS and re-apply boundary conditions if true
        """
        if assemble_rhs:
            self.assemble_rhs()

        self._ksp.solve(self._b.vector, self._x.vector)

    @property
    def x(self):
        return self._x

    def __del__(self):
        self._ksp.destroy()
        self._A.destroy()
        self._b.vector.destroy()
        self._x.vector.destroy()


class LumpedProject():
    """Projector using a lumped mass matrix"""
    __slots__ = ["_form", "_petsc_options", "_bcs"]

    def __init__(self):
        """

        """
        raise NotImplementedError
