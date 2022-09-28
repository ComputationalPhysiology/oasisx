# Copyright (C) 2022 Jørgen Schartum Dokken
#
# This file is part of Oasisx
# SPDX-License-Identifier:    MIT

from typing import List, Tuple

import numpy.typing as npt
import ufl
import numpy as np
from dolfinx import fem as _fem
from dolfinx import mesh as _mesh
from .ksp import KSPSolver
from petsc4py import PETSc as _PETSc


class FractionalStep_AB_CN():
    """
    Create the fractional step solver with Adam-Bashforth linearization
    of the convective term, and Crank-Nicholson time discretization.

    Args:
        mesh: The computational domain
        u_element: A tuple describing the finite element family and
            degree of the element used for the velocity
        p_element: A tuple describing the finite element family and
            degree of the element used for the pressure
        solver_options: Dictionary with keys 'tentative', 'pressure' and 'scalar',
            where each key leads to a dictionary of of PETSc options for each problem
        jit_options: Just in time parameters, to optimize form compilation
    """
    _mesh: _mesh.Mesh  # The computational domain

    # Velocity component and mapping to parent space
    _Vi: List[Tuple[_fem.FunctionSpace, npt.NDArray[np.int32]]]
    _V: _fem.FunctionSpace  # Velocity function space
    _Q: _fem.FunctionSpace  # Pressure function space

    _u: List[_fem.Function]  # Velocity at time t
    _u1: List[_fem.Function]  # Velocity at time t - dt
    _u2: List[_fem.Function]  # Velocity at time t - 2*dt
    _uab: List[_fem.Function]  # Explicit part of Adam Bashforth convection term
    _p: _fem.Function  # Pressure at time t
    _p1: _fem.Function  # Pressure at time t - dt
    _dp: _fem.Function  # Pressure correction

    _solver_u: KSPSolver
    _solver_p: KSPSolver
    _solver_c: KSPSolver

    _conv_Vi: _fem.FormMetaClass  # Compiled form of Adam Bashforth convection
    _mass_Vi: _fem.FormMetaClass  # Compiled form for mass matrix

    _stiffness_Vi: _fem.FormMetaClass  # Compiled form for stiffness matrix
    _stiffness_Q: _fem.FormMetaClass  # Compiled form for pressure Laplacian

    _p_rhs: _fem.Function  # Function for holding RHS of pressure correction
    _p_b: _fem.Function  # Function holding RHS vector for pressure correction

    _c_rhs: List[_fem.FormMetaClass]  # List of velocity update terms for each component

    _M: _PETSc.Mat  # Mass matrix
    _K: _PETSc.Mat  # Stiffnes matrix
    _Ap: _PETSc.Mat
    __slots__ = tuple(__annotations__)

    def __init__(self, mesh: _mesh.Mesh, u_element: Tuple[str, int],
                 p_element: Tuple[str, int], solver_options: dict = None,
                 jit_options: dict = None):
        v_el = ufl.VectorElement(u_element[0], mesh.ufl_cell(), u_element[1])
        p_el = ufl.FiniteElement(p_element[0], mesh.ufl_cell(), p_element[1])

        # Initialize velocity functions for variational problem
        self._V = _fem.FunctionSpace(mesh, v_el)
        self._Vi = [self._V.sub(i).collapse() for i in range(self._V.num_sub_spaces)]
        self._u = [_fem.Function(Vi[0], name=f"u_{i}") for (i, Vi) in enumerate(self._Vi)]
        self._u1 = [_fem.Function(Vi[0], name=f"u_{i}^1") for (i, Vi) in enumerate(self._Vi)]
        self._u2 = [_fem.Function(Vi[0], name=f"u_{i}^2") for (i, Vi) in enumerate(self._Vi)]
        self._uab = [_fem.Function(Vi[0], name=f"u_{i}^ab") for (i, Vi) in enumerate(self._Vi)]

        # Initialize pressure functions for varitional problem
        self._Q = _fem.FunctionSpace(mesh, p_el)
        self._p = _fem.Function(self._Q)
        self._dp = _fem.Function(self._Q)

        # Create solvers for each step
        self._solver_u = KSPSolver(mesh.comm, solver_options.get("tentative"))
        self._solver_p = KSPSolver(mesh.comm, solver_options.get("pressure"))
        self._solver_c = KSPSolver(mesh.comm, solver_options.get("scalar"))

        # Precompile forms and allocate matrices
        jit_params = {} if jit_params is None else jit_params
        self._compile_forms(jit_params)

        # Assemble matrices

        # Allocate extra work matrices

    def _compile_forms(self, jit_params: dict):
        dx = ufl.Measure("dx", domain=self._mesh)
        u = ufl.TrialFunction(self._Vi[0][0])
        v = ufl.TestFunction(self._Vi[0][0])

        # Mass matrix for velocity component
        self._mass_Vi = _fem.form(u*v*dx, jit_params=jit_params)
        self._M = _fem.petsc.create_matrix(self._mass_Vi)

        # Stiffness matrix for velocity component
        self._stiffness_Vi = _fem.form(ufl.inner(ufl.grad(u), ufl.grad(v))*dx,
                                       jit_params=jit_params)
        self._K = _fem.petsc.create_matrix(self._stiffness_Vi)

        # Pressure Laplacian/stiffness matrix
        p = ufl.TrialFunction(self._Q)
        q = ufl.TestFunction(self._Q)
        self._stiffness_Q = _fem.form(ufl.inner(ufl.grad(p), ufl.grad(q))*ufl.dx,
                                      jit_params=jit_params)
        self._Ap = _fem.petsc.create_matrix(self._stiffness_Q)

        # RHS for pressure correction (unscaled)
        self._p_rhs = _fem.form(ufl.div(ufl.as_vector(self._u))*q*dx, jit_params=jit_params)
        self._p_b = _fem.Function(self._Q)

        # RHS for velocity update
        self._c_rhs = [_fem.form(self._p.dx(i) * v*dx, jit_params=jit_params)
                       for i in range(len(self._Vi))]

        # Convection term for Adams Bashforth step
        self._conv_Vi = _fem.form(ufl.inner(v, ufl.dot(self._uab, ufl.nabla_grad(u)))*dx,
                                  jit_params=jit_params)

        # NOTE: Add a_scalar when needed (see cylinder example)
