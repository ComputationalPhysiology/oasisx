# Copyright (C) 2022 Jørgen Schartum Dokken
#
# This file is part of Oasisx
# SPDX-License-Identifier:    MIT

from typing import List, Tuple, Optional

import numpy.typing as npt
import ufl
import numpy as np
from dolfinx import cpp as _cpp
from dolfinx import fem as _fem, la as _la
from dolfinx import mesh as _dmesh
from .ksp import KSPSolver
from petsc4py import PETSc as _PETSc


__all__ = ["FractionalStep_AB_CN"]


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
    _mesh: _dmesh.Mesh  # The computational domain

    # Velocity component and mapping to parent space
    _Vi: List[Tuple[_fem.FunctionSpace, npt.NDArray[np.int32]]]
    _V: _fem.FunctionSpace  # Velocity function space
    _Q: _fem.FunctionSpace  # Pressure function space

    _u: List[_fem.Function]  # Velocity at time t
    _u1: List[_fem.Function]  # Velocity at time t - dt
    _u2: List[_fem.Function]  # Velocity at time t - 2*dt
    _uab: List[_fem.Function]  # Explicit part of Adam Bashforth convection term
    _ps: _fem.Function  # Pressure at time t - 3/2 dt
    _p: _fem.Function  # Pressure at time t - 1/2 dt
    _dp: _fem.Function  # Pressure correction

    _solver_u: KSPSolver
    _solver_p: KSPSolver
    _solver_c: KSPSolver

    _conv_Vi: _fem.FormMetaClass  # Compiled form of Adam Bashforth convection
    _mass_Vi: _fem.FormMetaClass  # Compiled form for mass matrix
    # Compiled form of the pressure gradient for RHS of tentative velocity
    _grad_p: List[_fem.FormMetaClass]

    _stiffness_Vi: _fem.FormMetaClass  # Compiled form for stiffness matrix
    _stiffness_Q: _fem.FormMetaClass  # Compiled form for pressure Laplacian

    _b_tmp: List[_fem.Function]  # Temporary storage for tentative velocity
    _b0: List[_fem.Function]  # Constant body forces
    _b_u: List[_fem.Function]  # RHS of tentative velocity step
    _b_matvec: _fem.Function  # Temporary work array for tentative velocity rhs

    _b_c: _fem.Function  # Function for holding RHS of pressure correction

    _c_rhs: List[_fem.FormMetaClass]  # List of velocity update terms for each component
    _p_rhs: _fem.FormMetaClass
    _p_b: _fem.Function

    _M: _PETSc.Mat  # Mass matrix
    _K: _PETSc.Mat  # Stiffnes matrix
    _Ap: _PETSc.Mat  # Pressure Laplacian
    _A: _PETSc.Mat  # Matrix for tentative velocity step

    _bc_u: List[List[_fem.DirichletBCMetaClass]]
    _bc_p: List[_fem.DirichletBCMetaClass]

    _body_force: List[_fem.FormMetaClass]
    __slots__ = tuple(__annotations__)

    def __init__(self, mesh: _dmesh.Mesh, u_element: Tuple[str, int],
                 p_element: Tuple[str, int], solver_options: dict = None,
                 jit_options: dict = None, body_force: Optional[ufl.core.expr.Expr] = None):
        self._mesh = mesh

        v_el = ufl.VectorElement(u_element[0], mesh.ufl_cell(), u_element[1])
        p_el = ufl.FiniteElement(p_element[0], mesh.ufl_cell(), p_element[1])

        # Initialize velocity functions for variational problem
        self._V = _fem.FunctionSpace(mesh, v_el)
        self._Vi = [self._V.sub(i).collapse() for i in range(self._V.num_sub_spaces)]
        self._u = [_fem.Function(Vi[0], name=f"u{i}") for (i, Vi) in enumerate(self._Vi)]
        self._u1 = [_fem.Function(Vi[0], name=f"u_{i}1") for (i, Vi) in enumerate(self._Vi)]
        self._u2 = [_fem.Function(Vi[0], name=f"u_{i}2") for (i, Vi) in enumerate(self._Vi)]
        self._uab = [_fem.Function(Vi[0], name=f"u_{i}ab") for (i, Vi) in enumerate(self._Vi)]

        # RHS arrays
        self._b_u = [_fem.Function(Vi[0]) for (i, Vi) in enumerate(
            self._Vi)]
        self._b_tmp = [_fem.Function(Vi[0]) for (i, Vi) in enumerate(
            self._Vi)]
        self._b0 = [_fem.Function(Vi[0]) for (i, Vi) in enumerate(
            self._Vi)]
        self._b_matvec = _fem.Function(self._Vi[0][0])

        # Initialize pressure functions for varitional problem
        self._Q = _fem.FunctionSpace(mesh, p_el)
        self._ps = _fem.Function(self._Q)
        self._p = _fem.Function(self.Q)
        self._dp = _fem.Function(self._Q)

        # Create solvers for each step
        solver_options = {} if solver_options is None else solver_options
        self._solver_u = KSPSolver(mesh.comm, solver_options.get("tentative"))
        self._solver_p = KSPSolver(mesh.comm, solver_options.get("pressure"))
        self._solver_c = KSPSolver(mesh.comm, solver_options.get("scalar"))

        # Precompile forms and allocate matrices
        jit_options = {} if jit_options is None else jit_options
        if body_force is None:
            body_force = (0.,) * mesh.geometry.dim
        self._compile_and_allocate_forms(body_force, jit_options)

        # Analyze boundary conditions, bc_u, bc_p. Should probably be a dictionary with
        # bcs = {"u":[(Method, locator, interpolant)], "p": [(Method, locator, interpolant)]}
        # where method is an enum (topological, geometrical), and locator and
        # interpolant are lambda functions
        self._bc_u = [[]*self._mesh.geometry.dim]
        self._bc_p = []

        # Assemble constant matrices
        self._preassemble()

    def _compile_and_allocate_forms(self, body_force: ufl.core.expr.Expr,
                                    jit_options: dict):
        dx = ufl.Measure("dx", domain=self._mesh)
        u = ufl.TrialFunction(self._Vi[0][0])
        v = ufl.TestFunction(self._Vi[0][0])

        # -----------------Tentative velocity step----------------------
        self._body_force = []
        for force in body_force:
            try:
                force = _fem.Constant(self._mesh, force)
            except RuntimeError:
                pass
            self._body_force.append(_fem.form(force*v*dx, jit_params=jit_options))

        # Mass matrix for velocity component
        self._mass_Vi = _fem.form(u*v*dx, jit_params=jit_options)
        self._M = _fem.petsc.create_matrix(self._mass_Vi)

        self._A = _fem.petsc.create_matrix(self._mass_Vi)

        # Stiffness matrix for velocity component
        self._stiffness_Vi = _fem.form(ufl.inner(ufl.grad(u), ufl.grad(v))*dx,
                                       jit_params=jit_options)
        self._K = _fem.petsc.create_matrix(self._stiffness_Vi)

        # Pressure gradients
        self._grad_p = [_fem.form(self._ps.dx(i)*v[i]*dx, jit_params=jit_options)
                        for i in range(self._mesh.geometry.dim)]

        # -----------------Pressure currection step----------------------

        # Pressure Laplacian/stiffness matrix
        p = ufl.TrialFunction(self._Q)
        q = ufl.TestFunction(self._Q)
        self._stiffness_Q = _fem.form(ufl.inner(ufl.grad(p), ufl.grad(q))*ufl.dx,
                                      jit_params=jit_options)
        self._Ap = _fem.petsc.create_matrix(self._stiffness_Q)

        # RHS for pressure correction (unscaled)
        self._p_rhs = _fem.form(ufl.div(ufl.as_vector(self._u))*q*dx, jit_params=jit_options)
        self._p_b = _fem.Function(self._Q)

        # RHS for velocity update
        self._c_rhs = [_fem.form(self._p.dx(i) * v*dx, jit_params=jit_options)
                       for i in range(len(self._Vi))]

        # Convection term for Adams Bashforth step
        self._conv_Vi = _fem.form(ufl.inner(ufl.dot(ufl.as_vector(self._uab),
                                                    ufl.nabla_grad(u)), v)*dx,
                                  jit_params=jit_options)

    def _preassemble(self):
        _fem.petsc.assemble_matrix(self._M, self._mass_Vi)
        self._M.assemble()
        _fem.petsc.assemble_matrix(self._K, self._stiffness_Vi)
        self._K.assemble()
        _fem.petsc.assemble_matrix(self._Ap, self._stiffness_Q)
        self._Ap.assemble()

        # Assemble constant body forces
        for i in range(len(self._u)):
            self._b_tmp[i].x.set(0.0)
            _fem.petsc.assemble_vector(self._b_tmp[i].vector, self._body_force[i])

    def assemble_first(self, dt: float, nu: float):
        """
        Assembly routine for first iteration of pressure/velocity system.

        .. math::
            A=\\frac{1}{\\delta t}M  + \\frac{1}{2} C +\\frac{1}{2}\\nu K

        where `M` is the mass matrix, `K` the stiffness matrix and `C` the convective term.
        We also assemble parts of the right hand side:

        .. math::

            b_k=\\left(\\frac{1}{\\delta t}M  + \\frac{1}{2} C +\\frac{1}{2}\\nu K\\right)u_k^{n-1}
            + \\int_{\\Omega} f_k^{n-\\frac{1}{2}}v_k~\\mathrm{d}x

        Args:
            dt: The time step
            nu: The kinematic viscosity
        """
        self._u1[0].x.array[:] = 2
        # Update explicit part of Adam-Bashforth approximation with previous time step
        for i, (u_ab, u_1, u_2) in enumerate(zip(self._uab, self._u1, self._u2)):
            u_ab.x.set(0)
            u_ab.x.array[:] = 1.5 * u_1.x.array - 0.5 * u_2.x.array

        self._A.zeroEntries()
        _fem.petsc.assemble_matrix(A=self._A, a=self._conv_Vi)  # type: ignore
        self._A.assemble()
        self._A.scale(-0.5)  # Negative convection on the rhs
        self._A.axpy(1./dt, self._M, _PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN)

        # Add diffusion
        self._A.axpy(-0.5*nu, self._K, _PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN)

        # Compute rhs for all velocity components
        for i, ui in enumerate(self._u):
            # Start with body force
            self._b_tmp[i].x.array[:] = self._b0[i].x.array[:]

            # Add transient convection and difffusion
            self._A.mult(self._u1[i].vector, self._b_matvec.vector)
            self._b_matvec.x.scatter_reverse(_la.ScatterMode.add)
            self._b_matvec.x.scatter_forward()
            self._b_tmp[i].x.array[:] += self._b_matvec.x.array[:]

        # Reset matrix for lhs
        self._A.scale(-1)
        self._A.axpy(2./dt, self._M,  _PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN)
        # NOTE: This would not work if we have different DirichletBCs on different components
        for bc in self._bc_u[0]:
            self._A.zeroRowsLocal(bc.dof_indices()[0], 1.)

    def velocity_tentative_assemble(self):
        """
        Assemble RHS of tentative velocity equation computing the :math:`p^*`-dependent term of

        .. math::

            b_k \\mathrel{+}= b(u_k^{n-1}) + \\int_{\\Omega} 
            \\frac{\\partial p^*}{\\partial x_k}v_k~\\mathrm{d}x.

        :math:`b(u_k^{n-1})` is computed in `assemble_first`.
        """

        pass
