# Copyright (C) 2022 Jørgen Schartum Dokken
#
# This file is part of Oasisx
# SPDX-License-Identifier:    MIT

from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import ufl
from dolfinx import cpp as _cpp
from dolfinx import fem as _fem
from dolfinx import la as _la
from dolfinx import mesh as _dmesh
from petsc4py import PETSc as _PETSc

from .bcs import DirichletBC
from .ksp import KSPSolver

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
        bcs_u: List of Dirichlet BCs for each component of the velocity
        bcs_p: List of Dirichlet BCs for the pressure
        solver_options: Dictionary with keys `'tentative'`, `'pressure'` and `'scalar'`,
            where each key leads to a dictionary of of PETSc options for each problem
        jit_options: Just in time parameters, to optimize form compilation
        options: Options for the Oasis solver.
            `"low_memory_version"` `True`/`False` changes if
            :math:`\\int\\nabla_k p^* v~\\mathrm{d}x` is assembled as
            `True`: directly into a vector or `False`: matrix-vector product.
            Default value: True
            '"bc_topological"` `True`/`False`. changes how the Dirichlet dofs are located.
            If True `facet_markers` has to be supplied.
        body_force: List of forces acting in each direction (x,y,z)
    """

    # -----------------------Multi-step variables-------------------------------
    _mesh: _dmesh.Mesh  # The computational domain

    # Velocity component and mapping to parent space
    _Vi: List[Tuple[_fem.FunctionSpace, npt.NDArray[np.int32]]]
    _V: _fem.FunctionSpace  # Velocity function space
    _Q: _fem.FunctionSpace  # Pressure function space

    _mass_Vi: _fem.FormMetaClass  # Compiled form for mass matrix
    _M: _PETSc.Mat  # Mass matrix
    _bcs_u: List[List[DirichletBC]]
    _bcs_p: List[DirichletBC]

    # -----------------------Tentative velocity step----------------------------
    _u: List[_fem.Function]  # Velocity at time t
    _u1: List[_fem.Function]  # Velocity at time t - dt
    _u2: List[_fem.Function]  # Velocity at time t - 2*dt
    _uab: List[_fem.Function]  # Explicit part of Adam Bashforth convection term
    _ps: _fem.Function  # Pressure at time t - 3/2 dt
    _sol_u: _fem.Function  # Tentative velocity as vector function (for outputting)
    _solver_u: KSPSolver

    _conv_Vi: _fem.FormMetaClass  # Compiled form of Adam Bashforth convection
    # Compiled form of the pressure gradient for RHS of tentative velocity
    _grad_p: List[_fem.FormMetaClass]
    _stiffness_Vi: _fem.FormMetaClass  # Compiled form for stiffness matrix
    _body_force: List[_fem.FormMetaClass]

    _b_tmp: List[_fem.Function]  # Working array for previous time step of tenative velocity RHS
    _b0: List[_fem.Function]  # Working array for constant body force contribution
    _b_u: List[_fem.Function]  # RHS of tentative velocity step
    _b_matvec: _fem.Function  # Work array for tentative velocity matrix vector product
    _b_gp: List[_fem.Function]  # Work array for gradient p term on RHS

    _A: _PETSc.Mat  # Matrix for tentative velocity step
    _grad_p_M: List[_PETSc.Mat]  # Matrix for grad-p operator
    _K: _PETSc.Mat  # Stiffness matrix

    # Indicating if grad(p)*v*dx and div(u)*q*dx term is assembled as vector or matrix-vector product
    _low_memory: bool

    # ----------------------Pressure correction---------------------------------
    _p: _fem.Function  # Pressure at time t - 1/2 dt
    _dp: _fem.Function  # Pressure correction
    _b_c: _fem.Function  # Function for holding RHS of pressure correction
    _same_space: bool  # Indicator if V and Q are the same

    _solver_p: KSPSolver
    _stiffness_Q: _fem.FormMetaClass  # Compiled form for pressure Laplacian
    _p_rhs: List[_fem.FormMetaClass]  # List of rhs for pressure correction
    _p_M: List[_PETSc.Mat]  # Matrices for non-low memory version of RHS assembly
    _Ap: _PETSc.Mat  # Pressure Laplacian
    _p_tmp: _fem.Function  # Working vector for matrix vector    products in non-low memory version

    # ----------------------Velocity update-------------------------------------
    _solver_c: KSPSolver
    _c_rhs: List[_fem.FormMetaClass]  # List of velocity update terms for each component
    _c_M: List[_PETSc.Mat]  # List of matrices for mat-vec operation in velocity update
    _p_b: _fem.Function
    _tmp_update: _fem.Function
    __slots__ = tuple(__annotations__)

    def __init__(self, mesh: _dmesh.Mesh, u_element: Tuple[str, int],
                 p_element: Tuple[str, int], bcs_u: List[List[DirichletBC]],
                 bcs_p: List[DirichletBC],
                 solver_options: dict = None, jit_options: dict = None,
                 body_force: Optional[ufl.core.expr.Expr] = None,
                 options: dict = None):
        self._mesh = mesh

        v_el = ufl.VectorElement(u_element[0], mesh.ufl_cell(), u_element[1])
        p_el = ufl.FiniteElement(p_element[0], mesh.ufl_cell(), p_element[1])
        if v_el.extract_component(0)[1] == p_el:
            self._same_space = True

        # Initialize velocity functions for variational problem
        self._V = _fem.FunctionSpace(mesh, v_el)
        self._sol_u = _fem.Function(self._V)  # Function for outputting vector functions

        self._Vi = [self._V.sub(i).collapse() for i in range(self._V.num_sub_spaces)]
        self._u = [_fem.Function(Vi[0], name=f"u{i}") for (i, Vi) in enumerate(self._Vi)]
        self._u1 = [_fem.Function(Vi[0], name=f"u_{i}1") for (i, Vi) in enumerate(self._Vi)]
        self._u2 = [_fem.Function(Vi[0], name=f"u_{i}2") for (i, Vi) in enumerate(self._Vi)]
        self._uab = [_fem.Function(Vi[0], name=f"u_{i}ab") for (i, Vi) in enumerate(self._Vi)]

        # Create boundary conditons for velocity spaces
        self._bcs_u = bcs_u
        for bc_i, Vi in zip(self._bcs_u, self._Vi):
            for bc in bc_i:
                bc.create_bc(Vi[0])

        # Working arrays
        self._b_tmp = [_fem.Function(Vi[0]) for (i, Vi) in enumerate(
            self._Vi)]
        self._b_matvec = _fem.Function(self._Vi[0][0])
        self._b_gp = [_fem.Function(Vi[0]) for (i, Vi) in enumerate(
            self._Vi)]
        # RHS arrays
        self._b_u = [_fem.Function(Vi[0]) for (i, Vi) in enumerate(
            self._Vi)]
        self._b0 = [_fem.Function(Vi[0]) for (i, Vi) in enumerate(
            self._Vi)]

        # Initialize pressure functions for varitional problem
        self._Q = _fem.FunctionSpace(mesh, p_el)
        self._ps = _fem.Function(self._Q)  # Pressure used in tentative velocity scheme
        self._p = _fem.Function(self._Q)
        self._dp = _fem.Function(self._Q)
        self._b_c = _fem.Function(self._Q)

        # Create boundary conditions for pressure space
        self._bcs_p = bcs_p
        for bc in self._bcs_p:
            bc.create_bc(self._Q)

        # Create solvers for each step
        solver_options = {} if solver_options is None else solver_options
        self._solver_u = KSPSolver(mesh.comm, solver_options.get("tentative"))
        self._solver_p = KSPSolver(mesh.comm, solver_options.get("pressure"))
        self._solver_c = KSPSolver(mesh.comm, solver_options.get("scalar"))

        if options is None:
            options = {}
        self._low_memory = options.get("low_memory_version", True)

        # Precompile forms and allocate matrices
        jit_options = {} if jit_options is None else jit_options
        if body_force is None:
            body_force = (0.,) * mesh.geometry.dim
        self._compile_and_allocate_forms(body_force, jit_options)

        # Assemble constant matrices
        self._preassemble()

        # Set solver operator
        self._solver_p.setOperators(self._Ap)
        self._solver_p.setOptions(self._Ap)
        self._solver_c.setOperators(self._Ap)

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
        p = ufl.TrialFunction(self._Q)
        if self._low_memory:
            self._grad_p = [_fem.form(self._ps.dx(i)*v*dx, jit_params=jit_options)
                            for i in range(self._mesh.geometry.dim)]
        else:
            self._grad_p = [_fem.form(p.dx(i)*v*dx, jit_params=jit_options)
                            for i in range(self._mesh.geometry.dim)]
            self._grad_p_M = [_fem.petsc.create_matrix(grad_p) for grad_p in self._grad_p]

        # -----------------Pressure currection step----------------------

        # Pressure Laplacian/stiffness matrix
        q = ufl.TestFunction(self._Q)
        self._stiffness_Q = _fem.form(ufl.inner(ufl.grad(p), ufl.grad(q))*ufl.dx,
                                      jit_params=jit_options)
        self._Ap = _fem.petsc.create_matrix(self._stiffness_Q)

        # RHS for pressure correction (unscaled)
        if self._low_memory:
            self._p_rhs = [_fem.form(ufl.div(ufl.as_vector(self._u))*q*dx, jit_params=jit_options)]
        else:
            self._p_rhs = [_fem.form(u.dx(i)*q*dx, jit_params=jit_options)
                           for i in range(self._mesh.geometry.dim)]
            self._p_M = [_fem.petsc.create_matrix(rhs) for rhs in self._p_rhs]
            self._p_tmp = _fem.Function(self._Q)

        # ---------------------------Velocity update-----------------------
        # RHS for velocity update
        self._p_b = _fem.Function(self._Vi[0][0])
        self._tmp_update = _fem.Function(self._Vi[0][0])
        if self._low_memory:
            self._c_rhs = [_fem.form(self._dp.dx(i) * v*dx, jit_params=jit_options)
                           for i in range(len(self._Vi))]
        else:
            self._c_M = self._grad_p_M

        # Convection term for Adams Bashforth step
        self._conv_Vi = _fem.form(ufl.inner(ufl.dot(ufl.as_vector(self._uab),
                                                    ufl.nabla_grad(u)), v)*dx,
                                  jit_params=jit_options)

    def _preassemble(self):
        _fem.petsc.assemble_matrix(self._M, self._mass_Vi)
        self._M.assemble()
        _fem.petsc.assemble_matrix(self._K, self._stiffness_Vi)
        self._K.assemble()

        # Assemble stiffness matrix with boundary conditions
        _fem.petsc.assemble_matrix(self._Ap, self._stiffness_Q, bcs=[
            bc._bc for bc in self._bcs_p])
        self._Ap.assemble()

        if len(self._bcs_p) == 0:
            nullspace = _PETSc.NullSpace().create(constant=True, comm=self._mesh.comm)
            self._Ap.setNullSpace(nullspace)

        if not self._low_memory:
            for i in range(self._mesh.geometry.dim):
                # Preassemble pressure matrix for tentantive velocity
                _fem.petsc.assemble_matrix(self._grad_p_M[i], self._grad_p[i])
                self._grad_p_M[i].assemble()

                # Assemble pressure RHS matrices
                _fem.petsc.assemble_matrix(self._p_M[i], self._p_rhs[i])
                self._p_M[i].assemble()

        # Assemble constant body forces
        for i in range(len(self._u)):
            self._b0[i].x.set(0.0)
            _fem.petsc.assemble_vector(self._b0[i].vector, self._body_force[i])
            self._b0[i].x.scatter_reverse(_la.ScatterMode.add)

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
        # Update explicit part of Adam-Bashforth approximation with previous time step
        for i, (u_ab, u_1, u_2) in enumerate(zip(self._uab, self._u1, self._u2)):
            u_ab.x.set(0)
            u_ab.x.array[:] = 1.5 * u_1.x.array - 0.5 * u_2.x.array

        self._A.zeroEntries()
        _fem.petsc.assemble_matrix(self._A, a=self._conv_Vi)  # type: ignore
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
            self._b_matvec.x.scatter_forward()
            self._b_tmp[i].x.array[:] += self._b_matvec.x.array[:]

        # Reset matrix for lhs
        self._A.scale(-1)
        self._A.axpy(2./dt, self._M,  _PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN)
        # NOTE: This would not work if we have different DirichletBCs on different components
        for bc in self._bcs_u[0]:
            self._A.zeroRowsLocal(bc._bc.dof_indices()[0], 1.)  # type: ignore

    def velocity_tentative_assemble(self):
        """
        Assemble RHS of tentative velocity equation computing the :math:`p^*`-dependent term of

        .. math::

            b_k \\mathrel{+}= b(u_k^{n-1}) + \\int_{\\Omega}
            \\frac{\\partial p^*}{\\partial x_k}v_k~\\mathrm{d}x.

        :math:`b(u_k^{n-1})` is computed in :py:obj:`FractionalStep_AB_CN.assemble_first`.
        """
        if self._low_memory:
            # Using the fact that all the gradient forms has the same coefficient
            coeffs = _cpp.fem.pack_coefficients(self._grad_p[0])
            for i in range(self._mesh.geometry.dim):
                self._b_gp[i].x.set(0.)
                _fem.petsc.assemble_vector(
                    self._b_gp[i].vector, self._grad_p[i], coeffs=coeffs, constants=[])
                self._b_gp[i].x.scatter_reverse(_la.ScatterMode.add)
                self._b_gp[i].x.scatter_forward()
        else:
            for i in range(self._mesh.geometry.dim):
                self._b_gp[i].x.set(0.)
                self._grad_p_M[i].mult(self._ps.vector, self._b_gp[i].vector)
                self._b_gp[i].x.scatter_forward()
        # Update RHS
        for i in range(self._mesh.geometry.dim):
            self._b_u[i].x.array[:] = self._b_tmp[i].x.array - self._b_gp[i].x.array

    def velocity_tentative_solve(self) -> Tuple[float, npt.NDArray[np.int32]]:
        """
        Apply Dirichlet boundary condition to RHS vector and solver linear algebra system

        Returns the difference between the two solutions and the solver error codes
        """
        self._solver_u.setOperators(self._A)
        self._solver_u.setOptions(self._A)
        diff = 0
        errors = np.zeros(self._mesh.geometry.dim, dtype=np.int32)
        for i in range(self._mesh.geometry.dim):
            for bc in self._bcs_u[i]:
                bc.apply(self._b_u[i].vector)
            # Use temporary storage, as it is only used in `assemble_first`
            self._u[i].vector.copy(result=self._b_matvec.vector)
            errors[i] = self._solver_u.solve(self._b_u[i].vector, self._u[i])

            # Compute difference from last inner iter
            self._b_matvec.vector.axpy(-1, self._u[i].vector)
            diff += self._b_matvec.vector.norm(_PETSc.NormType.NORM_2)
        return diff, errors

    def tenative_velocity(self, dt: float, nu: float, max_error: float = 1e-12,
                          max_iter: int = 10) -> float:
        """
        Propagate the tenative velocity by one step

        Args:
            dt: The time step
            nu: The kinematic velocity
            max_error: The maximal difference for inner iterations of solving `u`
            max_iter: Maximal number of inner iterations for `u`
        Returns:
            The difference between the two last inner iterations

        """
        inner_it = 0
        diff = 1e8
        self.assemble_first(dt, nu)
        while inner_it < max_iter and diff > max_error:
            inner_it += 1
            self.velocity_tentative_assemble()
            diff, errors = self.velocity_tentative_solve()
            assert (errors > 0).all()
        return diff

    def pressure_assemble(self, dt: float):
        """
        Assemble RHS for pressure correction term        

        .. math::

            b_c = \\int_{\\Omega} \\mathrm{div} u^l q~\\mathrm{d}x.

        """
        self._b_c.x.set(0.)
        if self._low_memory:
            _fem.petsc.assemble_vector(self._b_c.vector, self._p_rhs[0])
        else:
            for i in range(self._mesh.geometry.dim):
                self._p_M[i].mult(self._u[i].vector, self._p_tmp.vector)
                self._b_c.vector.axpy(-1/dt, self._p_tmp.vector)
        # Apply boundary conditions to the rhs
        bc_p = [bc._bc for bc in self._bcs['p']]
        self._b_c.x.scatter_reverse(_la.ScatterMode.add)
        _fem.petsc.set_bc(self._b_c.vector, bc_p)

        # Set pressure DirichletBC condition for time (t+dt/2)
        _fem.petsc.set_bc(self._b_c.vector, self._bcs_p)

        self._b_c.x.scatter_forward()

    def pressure_solve(self) -> np.int32:
        """
        Solve pressure correction problem 
        """

        # Set difference vector to previous time step
        self._dp.x.array[:] = -self._ps.x.array[:]

        if len(self._bcs_p) == 0:
            nullspace = self._Ap.getNullSpace()
            nullspace.remove(self._b_c.vector)
        converged = self._solver_p.solve(self._b_c.vector, self._p)

        # Compute phi  = p^(n-1/2) - p*
        self._dp.vector.axpy(1, self._p.vector)
        return converged

    def velocity_update(self, dt) -> npt.NDArray[np.int32]:
        """
        Compute Velocity update
        """
        errors = np.zeros(self._mesh.geometry.dim, dtype=np.int32)
        if self._low_memory:
            for i in range(self._mesh.geometry.dim):
                self._M.mult(self._u[i].vector, self._p_b.vector)
                self._tmp_update.x.set(0)
                _fem.petsc.assemble_vector(self._tmp_update.vector, self._c_rhs[i])
                self._tmp_update.x.scatter_reverse(_la.ScatterMode.add)
                self._p_b.vector.axpy(-dt, self._tmp_update.vector)
                errors[i] = self._solver_c.solve(self._p_b.vector, self._u[i])
        else:
            for i in range(self._mesh.geometry.dim):
                self._M.mult(self._u[i].vector, self._p_b.vector)
                self._tmp_update.x.set(0)
                self._c_M[i].mult(self._dp.vector, self._tmp_update.vector)
                self._p_b.vector.axpy(-1, self._tmp_update.vector)
                errors[i] = self._solver_c.solve(self._p_b.vector, self._u[i])

        return errors

    @property
    def u(self):
        """
        Return the solution to the tentative velocity equation as a vector function
        """
        for ui, (Vi, map) in zip(self._u, self._Vi):
            self._sol_u.x.array[map] = ui.x.array
        return self._sol_u
