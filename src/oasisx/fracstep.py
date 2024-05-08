# Copyright (C) 2022 Jørgen Schartum Dokken
#
# This file is part of Oasisx
# SPDX-License-Identifier:    MIT

import logging
from typing import List, Optional, Tuple, Union

from mpi4py import MPI as _MPI
from petsc4py import PETSc as _PETSc

import basix
import dolfinx.fem.petsc as _petsc
import numpy as np
import numpy.typing as npt
import ufl
from dolfinx import cpp as _cpp
from dolfinx import default_scalar_type
from dolfinx import fem as _fem
from dolfinx import la as _la
from dolfinx import mesh as _dmesh

from .bcs import DirichletBC, PressureBC
from .function import Projector
from .ksp import KSPSolver

__all__ = ["FractionalStep_AB_CN"]


class FractionalStep_AB_CN:
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
        rotational: If True, use rotational form of pressure update
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

    # Convenience functions
    _sol_u: _fem.Function  # Tentative velocity as vector function (for outputting)

    # Velocity component and mapping to parent space
    _Vi: List[Tuple[_fem.FunctionSpace, npt.NDArray[np.int32]]]
    _V: _fem.FunctionSpace  # Velocity function space
    _Q: _fem.FunctionSpace  # Pressure function space

    # Mass matrix for velocity component
    _mass_Vi: _fem.Form
    _M: _PETSc.Mat  # type: ignore

    # Boundary conditions
    _bcs_u: List[List[DirichletBC]]
    _bcs_p: List[PressureBC]

    # -----------------------Tentative velocity step----------------------------

    # Coefficients of velocity and pressure
    _u: List[_fem.Function]  # Velocity at time t
    _u1: List[_fem.Function]  # Velocity at time t - dt
    _u2: List[_fem.Function]  # Velocity at time t - 2*dt
    _ps: _fem.Function  # Tentative pressure

    # Linear algebra structures structures
    _A: _PETSc.Mat  # type: ignore
    _rhs1: List[_fem.Function]  # RHS for each component of the tentative velocity
    _solver_u: KSPSolver

    # Adams-Bashforth convection term
    _conv_Vi: _fem.Form
    _uab: List[_fem.Function]  # Explicit part of Adam Bashforth convection term

    # Stiffness matrix for velocity component
    _stiffness_Vi: _fem.Form
    _K: _PETSc.Mat  # type: ignore

    _p_vdxi: List[_fem.Form]  # Volume contributions of tentative pressure to RHS
    _p_vdxi_Mat: List[_PETSc.Mat]  # type: ignore
    # Low memory version
    _p_vdxi_Vec: List[_PETSc.Vec]  # type: ignore

    # Body forces
    _b0: List[_fem.Function]
    _body_force: List[_fem.Form]
    _b_first: List[_fem.Function]  # RHS consisting of all variables from previous time step
    _p_surf: List[_fem.Form]  # Surface terms for pressure at outlets at t-1/2

    # Working arrays
    _wrk_vel: List[_fem.Function]  # Working arrays for velocity space
    _wrk_comp: _fem.Function

    # Indicating if grad(p)*v*dx and div(u)*q*dx term is assembled as
    # vector or matrix-vector product
    _low_memory: bool

    # ----------------------Pressure correction---------------------------------
    _p: _fem.Function  # Pressure at time t - 1/2 dt
    _dp: _fem.Function  # Pressure correction
    _b2: _fem.Function  # Function for holding RHS of pressure correction

    _solver_p: KSPSolver
    _stiffness_Q: _fem.Form  # Compiled form for pressure Laplacian
    _p_rhs: List[_fem.Form]  # List of rhs for pressure correction
    # Matrices for non-low memory version of RHS assembly
    _divu_Mat: List[_PETSc.Mat]  # type: ignore
    # Pressure Laplacian
    _Ap: _PETSc.Mat  # type: ignore
    _wrk_p: _fem.Function  # Working vector for matrix vector    products in non-low memory version

    # Rotational pressure correction variables
    _projector_p: Optional[Projector]
    _xi: Optional[_fem.Constant]
    _nu: Optional[_fem.Constant]

    # ----------------------Velocity update-------------------------------------
    _solver_c: KSPSolver

    # grad_i(phi) v_i operator
    _grad_p: List[_fem.Form]
    _grad_p_Mat: List[_PETSc.Mat]  # type: ignore
    # Low memory version
    _grad_p_Vec: List[_PETSc.Vec]  # type: ignore
    _b3: _fem.Function
    # Mass matrix with bcs applied
    _M_bcs: _PETSc.Mat  # type: ignore

    # Annotate all functions
    # __slots__ = tuple(__annotations__)

    def __init__(
        self,
        mesh: _dmesh.Mesh,
        u_element: Union[Tuple[str, int], basix.finite_element.FiniteElement],
        p_element: Union[Tuple[str, int], basix.finite_element.FiniteElement],
        bcs_u: List[List[DirichletBC]],
        bcs_p: List[PressureBC],
        rotational: bool = False,
        solver_options: Optional[dict] = None,
        jit_options: Optional[dict] = None,
        body_force: Optional[ufl.core.expr.Expr] = None,
        options: Optional[dict] = None,
    ):
        self._mesh = mesh
        cellname = mesh.ufl_cell().cellname()
        try:
            v_family = basix.finite_element.string_to_family(u_element[0], cellname)  # type: ignore
            v_el = basix.ufl.element(
                v_family,
                cellname,
                u_element[1],  # type: ignore
                basix.LagrangeVariant.gll_warped,
                shape=(mesh.geometry.dim,),
            )
        except TypeError:
            v_el = u_element  # type: ignore
        try:
            p_family = basix.finite_element.string_to_family(p_element[0], cellname)  # type: ignore
            p_el = basix.ufl.element(
                p_family,
                cellname,
                p_element[1],  # type: ignore
                basix.LagrangeVariant.gll_warped,
            )
        except TypeError:
            p_el = p_element  # type: ignore

        # Initialize velocity functions for variational problem
        self._V = _fem.functionspace(mesh, v_el)
        self._sol_u = _fem.Function(self._V, name="u")  # Function for outputting vector functions

        self._Vi = [self._V.sub(i).collapse() for i in range(self._V.num_sub_spaces)]
        self._u = [_fem.Function(Vi[0], name=f"u{i}") for (i, Vi) in enumerate(self._Vi)]
        self._u1 = [_fem.Function(Vi[0], name=f"u_{i}1") for (i, Vi) in enumerate(self._Vi)]
        self._u2 = [_fem.Function(Vi[0], name=f"u_{i}2") for (i, Vi) in enumerate(self._Vi)]
        self._uab = [_fem.Function(Vi[0], name=f"u_{i}ab") for (i, Vi) in enumerate(self._Vi)]

        # Create boundary conditions for velocity spaces
        self._bcs_u = bcs_u
        for bc_i, Vi in zip(self._bcs_u, self._Vi):
            for bc in bc_i:
                bc.create_bc(Vi[0])

        # Working arrays
        self._wrk_vel = [_fem.Function(Vi[0]) for Vi in self._Vi]
        self._wrk_comp = _fem.Function(self._Vi[0][0])

        # RHS arrays
        self._rhs1 = [_fem.Function(Vi[0]) for Vi in self._Vi]
        self._b0 = [_fem.Function(Vi[0]) for Vi in self._Vi]
        self._b_first = [_fem.Function(Vi[0]) for Vi in self._Vi]

        # Initialize pressure functions for variational problem
        self._Q = _fem.functionspace(mesh, p_el)
        self._ps = _fem.Function(self._Q)  # Pressure used in tentative velocity scheme
        self._p = _fem.Function(self._Q)
        self._dp = _fem.Function(self._Q)
        self._b2 = _fem.Function(self._Q)

        # Create boundary conditions for pressure space
        forms: List[List[ufl.form.Form]] = [[] for _ in range(self._mesh.geometry.dim)]
        self._bcs_p = bcs_p
        for bcp in self._bcs_p:
            bcp.create_bcs(self._Vi[0][0], self._Q)
            for i in range(self._mesh.geometry.dim):
                forms[i].append(bcp.rhs(i))

        if len(self._bcs_p) > 0:
            self._p_surf = [_fem.form(sum(form)) for form in forms]

        # Create solvers for each step
        solver_options = {} if solver_options is None else solver_options
        self._solver_u = KSPSolver(
            mesh.comm, solver_options.get("tentative"), prefix="tentative_velocity"
        )
        self._solver_p = KSPSolver(
            mesh.comm, solver_options.get("pressure"), prefix="pressure_correction"
        )
        if rotational:
            self._xi = _fem.Constant(mesh, default_scalar_type(0.5))
            self._nu = _fem.Constant(mesh, default_scalar_type(1))
            update_expr = self._p + self._dp - self._xi * self._nu * ufl.div(ufl.as_vector(self._u))
            self._projector_p = Projector(
                update_expr,
                self._Q,
                bcs=[],
                petsc_options=solver_options.get("scalar"),
                jit_options=jit_options,
            )
        else:
            self._projector_p = None
            self._xi = None
            self._nu = None

        self._solver_c = KSPSolver(
            mesh.comm, solver_options.get("scalar"), prefix="velocity_update"
        )

        if options is None:
            options = {}
        self._low_memory = options.get("low_memory_version", True)

        # Precompile forms and allocate matrices
        jit_options = {} if jit_options is None else jit_options
        if body_force is None:
            body_force = (0.0,) * mesh.geometry.dim
        self._compile_and_allocate_forms(body_force, jit_options)

        # Assemble constant matrices
        self._preassemble()

        # Set solver operator
        self._solver_p.setOperators(self._Ap)
        self._solver_p.setOptions(self._Ap)
        self._solver_c.setOperators(self._M)
        self._solver_u.setOperators(self._A)
        self._solver_u.setOptions(self._A)

    def _compile_and_allocate_forms(self, body_force: ufl.core.expr.Expr, jit_options: dict):
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
            self._body_force.append(_fem.form(force * v * dx, jit_options=jit_options))

        # Mass matrix for velocity component
        self._mass_Vi = _fem.form(u * v * dx, jit_options=jit_options)
        self._M = _petsc.create_matrix(self._mass_Vi)
        self._A = _petsc.create_matrix(self._mass_Vi)

        # Stiffness matrix for velocity component
        self._stiffness_Vi = _fem.form(
            ufl.inner(ufl.grad(u), ufl.grad(v)) * dx, jit_options=jit_options
        )
        self._K = _petsc.create_matrix(self._stiffness_Vi)

        # Pressure contribution
        p = ufl.TrialFunction(self._Q)
        self._p_vdxi_Vec = [_fem.Function(Vi[0]) for Vi in self._Vi]
        if self._low_memory:
            self._p_vdxi = [
                _fem.form(self._ps * v.dx(i) * dx, jit_options=jit_options)
                for i in range(self._mesh.geometry.dim)
            ]
        else:
            self._p_vdxi = [
                _fem.form(p * v.dx(i) * dx, jit_options=jit_options)
                for i in range(self._mesh.geometry.dim)
            ]
            self._p_vdxi_Mat = [_petsc.create_matrix(grad_p) for grad_p in self._p_vdxi]

        # -----------------Pressure correction step----------------------

        # Pressure Laplacian/stiffness matrix
        q = ufl.TestFunction(self._Q)
        self._stiffness_Q = _fem.form(
            ufl.inner(ufl.grad(p), ufl.grad(q)) * ufl.dx, jit_options=jit_options
        )
        self._Ap = _petsc.create_matrix(self._stiffness_Q)

        # RHS for pressure correction (unscaled)
        if self._low_memory:
            self._p_rhs = [
                _fem.form(ufl.div(ufl.as_vector(self._u)) * q * dx, jit_options=jit_options)
            ]
        else:
            self._p_rhs = [
                _fem.form(u.dx(i) * q * dx, jit_options=jit_options)
                for i in range(self._mesh.geometry.dim)
            ]
            self._divu_Mat = [_petsc.create_matrix(rhs) for rhs in self._p_rhs]
            self._wrk_p = _fem.Function(self._Q)

        # ---------------------------Velocity update-----------------------
        # RHS for velocity update
        self._b3 = _fem.Function(self._Vi[0][0])
        if self._low_memory:
            self._grad_p = [
                _fem.form(self._dp.dx(i) * v * dx, jit_options=jit_options)
                for i in range(len(self._Vi))
            ]
        else:
            self._grad_p = [
                _fem.form(p.dx(i) * v * dx, jit_options=jit_options)
                for i in range(self._mesh.geometry.dim)
            ]
            self._grad_p_Mat = [_petsc.create_matrix(grad_p) for grad_p in self._grad_p]

        # Convection term for Adams Bashforth step
        self._conv_Vi = _fem.form(
            ufl.inner(ufl.dot(ufl.as_vector(self._uab), ufl.nabla_grad(u)), v) * dx,
            jit_options=jit_options,
        )

    def _preassemble(self):
        """
        Assemble time independent matrices and vectors

        This includes:
        1. Mass matrix for a component of the velocity
        2. Stiffness matrix for a component of the velocity
        3. Mass matrix for the pressure (attaches constant nullspace if necessary)
        4. The time independent body forces
        5. Pressure contribution in tentative velocity eq, (if low memory is turned off)
        6. Divergence term in pressure correction
        7. Pressure contribution in velocity update (if low memory is turned of)
        """
        _petsc.assemble_matrix(self._M, self._mass_Vi)
        self._M.assemble()
        _petsc.assemble_matrix(self._K, self._stiffness_Vi)
        self._K.assemble()

        # Assemble stiffness matrix with boundary conditions
        _petsc.assemble_matrix(self._Ap, self._stiffness_Q, bcs=[bc._bc for bc in self._bcs_p])
        self._Ap.assemble()
        if len(self._bcs_p) == 0:
            nullspace = _PETSc.NullSpace().create(constant=True, comm=self._mesh.comm)
            self._Ap.setNullSpace(nullspace)
            self._Ap.setNearNullSpace(nullspace)

        # Assemble constant body forces
        for i in range(len(self._u)):
            self._b0[i].x.array[:] = 0.0
            _petsc.assemble_vector(self._b0[i].vector, self._body_force[i])
            self._b0[i].x.scatter_reverse(_la.InsertMode.add)

        if not self._low_memory:
            for i in range(self._mesh.geometry.dim):
                # Preassemble tentative velocity matrix
                _petsc.assemble_matrix(self._p_vdxi_Mat[i], self._p_vdxi[i])
                self._p_vdxi_Mat[i].assemble()

                # Assemble pressure RHS matrices
                _petsc.assemble_matrix(self._grad_p_Mat[i], self._grad_p[i])
                self._grad_p_Mat[i].assemble()

                # Preassemble u.dx(i)q
                _petsc.assemble_matrix(self._divu_Mat[i], self._p_rhs[i])
                self._divu_Mat[i].assemble()

        # Create mass matrix with symmetrically applied bcs
        self._M_bcs = self._M.copy()
        for bcu in self._bcs_u[0]:
            self._M_bcs.zeroRowsColumnsLocal(bcu._bc._cpp_object.dof_indices()[0], 1.0)  # type: ignore

    def assemble_first(self, dt: float, nu: float):
        """
        Assembly routine for first iteration of pressure/velocity system.

        .. math::
            A=\\frac{1}{\\delta t}M  + \\frac{1}{2} C +\\frac{1}{2}\\nu K

        where `M` is the mass matrix, `K` the stiffness matrix and `C` the convective term.
        We also assemble parts of the right hand side:

        .. math::

            b_k=\\left(\\frac{1}{\\delta t}M  0 \\frac{1}{2} C -\\frac{1}{2}\\nu K\\right)u_k^{n-1}
            + \\int_{\\Omega} f_k^{n-\\frac{1}{2}}v_k~\\mathrm{d}x
            + \\int h^{n-\\frac{1}{2}} n_k \\nabla_k v~\\mathrm{d}x

        Args:
            dt: The time step
            nu: The kinematic viscosity
        """
        # Update explicit part of Adam-Bashforth approximation with previous time step
        for i, (u_ab, u_1, u_2) in enumerate(zip(self._uab, self._u1, self._u2)):
            u_ab.x.array[:] = 0
            u_ab.x.array[:] = 1.5 * u_1.x.array - 0.5 * u_2.x.array
        self._A.zeroEntries()
        _petsc.assemble_matrix(self._A, a=self._conv_Vi)  # type: ignore
        self._A.assemble()
        self._A.scale(-0.5)  # Negative convection on the rhs
        self._A.axpy(1.0 / dt, self._M, _PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN)  # type: ignore

        # Add diffusion
        self._A.axpy(-0.5 * nu, self._K, _PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN)  # type: ignore

        # Update Pressure BC
        for bc in self._bcs_p:
            bc.update_bc()

        # Compute rhs for all velocity components
        for i, _ in enumerate(self._u):
            self._b_first[i].x.array[:] = 0
            # Start with transient convection and diffusion
            self._A.mult(self._u1[i].vector, self._wrk_comp.vector)
            self._wrk_comp.x.scatter_forward()

            # Add body force
            self._wrk_comp.x.array[:] += self._b0[i].x.array[:]

            self._b_first[i].x.array[:] += self._wrk_comp.x.array[:]

            # Add pressure contribution
            if hasattr(self, "_p_surf") and self._p_surf[i].rank == 1:  # type: ignore
                self._wrk_comp.x.array[:] = 0
                _petsc.assemble_vector(self._wrk_comp.vector, self._p_surf[i])
                self._wrk_comp.x.scatter_reverse(_la.InsertMode.add)
                self._b_first[i].x.array[:] += self._wrk_comp.x.array[:]

        # Reset matrix for lhs
        self._A.scale(-1)
        self._A.axpy(2.0 / dt, self._M, _PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN)  # type: ignore
        # NOTE: This would not work if we have different DirichletBCs on different components
        for bcu in self._bcs_u[0]:
            self._A.zeroRowsLocal(bcu._bc._cpp_object.dof_indices()[0], 1.0)  # type: ignore

    def velocity_tentative_assemble(self):
        """
        Assemble RHS of tentative velocity equation computing the :math:`p^*`-dependent term of

        .. math::

            b_k \\mathrel{+}= b(u_k^{n-1}) + \\int_{\\Omega}
            p^*\\frac{\\partial v_k}{\\partial x_k}~\\mathrm{d}x.

        :math:`b(u_k^{n-1})` is computed in :py:obj:`FractionalStep_AB_CN.assemble_first`.
        """
        if self._low_memory:
            # Using the fact that all the gradient forms has the same coefficient
            coeffs = _cpp.fem.pack_coefficients(self._p_vdxi[0]._cpp_object)
            for i in range(self._mesh.geometry.dim):
                self._p_vdxi_Vec[i].x.array[:] = 0.0
                _petsc.assemble_vector(
                    self._p_vdxi_Vec[i].vector,
                    self._p_vdxi[i],
                    coeffs=coeffs,
                    constants=np.empty(0, dtype=self._p_vdxi_Vec[i].x.array.dtype),
                )
                self._p_vdxi_Vec[i].x.scatter_reverse(_la.InsertMode.add)
                self._p_vdxi_Vec[i].x.scatter_forward()
        else:
            for i in range(self._mesh.geometry.dim):
                self._p_vdxi_Vec[i].x.array[:] = 0.0
                self._p_vdxi_Mat[i].mult(self._ps.vector, self._p_vdxi_Vec[i].vector)
                self._p_vdxi_Vec[i].x.scatter_forward()

        # Update RHS
        for i in range(self._mesh.geometry.dim):
            self._rhs1[i].x.array[:] = self._b_first[i].x.array + self._p_vdxi_Vec[i].x.array

    def velocity_tentative_solve(self) -> Tuple[float, npt.NDArray[np.int32]]:
        """
        Apply Dirichlet boundary condition to RHS vector and solver linear algebra system

        Returns the difference between the two solutions and the solver error codes
        """
        diff = 0
        errors = np.zeros(self._mesh.geometry.dim, dtype=np.int32)
        for i in range(self._mesh.geometry.dim):
            for bc in self._bcs_u[i]:
                bc.apply(self._rhs1[i].vector)

            self._u[i].vector.copy(result=self._wrk_comp.vector)
            errors[i] = self._solver_u.solve(self._rhs1[i].vector, self._u[i])
            # Compute difference from last inner iter
            self._wrk_comp.vector.axpy(-1, self._u[i].vector)
            diff += self._wrk_comp.vector.norm(_PETSc.NormType.NORM_2)  # type: ignore
        return diff, errors

    def pressure_assemble(self, dt: float):
        """
        Assemble RHS for pressure correction term

        .. math::

            b_c = \\int_{\\Omega} \\mathrm{div} u^l q~\\mathrm{d}x.

        """
        self._b2.x.array[:] = 0.0
        if self._low_memory:
            _petsc.assemble_vector(self._b2.vector, self._p_rhs[0])
        else:
            for i in range(self._mesh.geometry.dim):
                self._divu_Mat[i].mult(self._u[i].vector, self._wrk_p.vector)
                self._b2.vector.axpy(1, self._wrk_p.vector)

        # Apply boundary conditions to the rhs
        self._b2.x.scatter_reverse(_la.InsertMode.add)
        self._b2.vector.scale(-1 / dt)

        # Set homogenous Dirichlet boundary condition on pressure correction
        bc_p = [bc._bc for bc in self._bcs_p]
        _petsc.set_bc(self._b2.vector, bc_p)
        self._b2.x.scatter_forward()

    def pressure_solve(self, nu: Optional[float] = None, rotational: bool = False) -> np.int32:
        """
        Solve pressure correction problem
        """
        logger = logging.getLogger("oasisx")

        # Set difference vector to previous time step
        if len(self._bcs_p) == 0:
            # If pressure nullspace, use mumps to deal with singular matrix
            nullspace_options = {
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
                "mat_mumps_icntl_24": 1,
                "mat_mumps_icntl_25": 0,
                "ksp_error_if_not_converged": 1,
            }
            logger.debug(f"Updating PETSc options to {nullspace_options}")
            nullspace = self._Ap.getNullSpace()
            nullspace.remove(self._b2.vector)
            self._solver_p.updateOptions(nullspace_options)
            self._solver_p.setOptions(self._Ap)

        converged = self._solver_p.solve(self._b2.vector, self._dp)
        if len(self._bcs_p) == 0:
            logger.debug("Making sure that mean of phi is 0 with lack of pressure conditions")
            vol = self._mesh.comm.allreduce(
                _fem.assemble_scalar(_fem.form(1 * ufl.dx(domain=self._mesh))),
                op=_MPI.SUM,
            )
            phi_avg = (
                self._mesh.comm.allreduce(
                    _fem.assemble_scalar(_fem.form(self._dp * ufl.dx)), op=_MPI.SUM
                )
                / vol
            )
            self._dp.x.array[:] -= phi_avg

        if self._projector_p is not None:
            if nu is not None and self._nu is not None:
                self._nu.value = nu
            else:
                raise RuntimeWarning(
                    "Kinematic viscosity not set for rotational pressure correction"
                )
            error = self._projector_p.solve(assemble_rhs=True)
            assert error > 0
            self._ps.x.array[:] = self._projector_p.x.x.array[:]
        else:
            self._ps.x.array[:] = self._p.x.array[:] + self._dp.x.array
        return converged

    def velocity_update(self, dt) -> npt.NDArray[np.int32]:
        """
        Compute Velocity update
        """
        errors = np.zeros(self._mesh.geometry.dim, dtype=np.int32)
        if self._low_memory:
            for i in range(self._mesh.geometry.dim):
                # Compute M u^{n-1}
                self._M.mult(self._u[i].vector, self._b3.vector)

                self._wrk_comp.x.array[:] = 0.0
                _petsc.assemble_vector(self._wrk_comp.vector, self._grad_p[i])
                self._wrk_comp.x.scatter_reverse(_la.InsertMode.add)

                # Subtract
                self._b3.vector.axpy(-dt, self._wrk_comp.vector)

                # Set bcs
                # bcs_u = [bcu._bc for bcu in self._bcs_u[i]]
                # self._wrk_comp.x.array[:] = 0
                # _petsc.apply_lifting(self._wrk_comp.vector, [self._mass_Vi], [bcs_u])
                # self._wrk_comp.x.scatter_reverse(_la.InsertMode.add)

                # self._b3.vector.axpy(1, self._wrk_comp.vector)
                # _petsc.set_bc(self._b3.vector, bcs_u)
                self._b3.x.scatter_forward()

                errors[i] = self._solver_c.solve(self._b3.vector, self._u[i])
        else:
            for i in range(self._mesh.geometry.dim):
                # Compute M u^{n-1}
                self._M.mult(self._u[i].vector, self._b3.vector)

                # Compute dphi/dx_i vi dx
                self._wrk_comp.x.array[:] = 0.0
                self._grad_p_Mat[i].mult(self._dp.vector, self._wrk_comp.vector)

                # Subtract
                self._b3.vector.axpy(-dt, self._wrk_comp.vector)

                # Set bcs
                # bcs_u = [bcu._bc for bcu in self._bcs_u[i]]
                # self._wrk_comp.x.array[:] = 0.
                # _petsc.apply_lifting(self._wrk_comp.vector, [self._mass_Vi], [bcs_u])
                # self._wrk_comp.x.scatter_reverse(_la.InsertMode.add)

                # self._b3.vector.axpy(1, self._wrk_comp.vector)
                # _petsc.set_bc(self._b3.vector, bcs_u)
                self._b3.x.scatter_forward()
                errors[i] = self._solver_c.solve(self._b3.vector, self._u[i])

        return errors

    def solve(self, dt: float, nu: float, max_error: float = 1e-12, max_iter: int = 10):
        """
        Propagate splitting scheme one time step

        Args:
            dt: The time step
            nu: The kinematic velocity
            max_error: The maximal difference for inner iterations of solving `u`
            max_iter: Maximal number of inner iterations for `u`

        """
        inner_it = 0
        diff = 1e8
        self._ps.x.array[:] = self._p.x.array[:]

        [[bc.update_bc() for bc in bcu] for bcu in self._bcs_u]
        self.assemble_first(dt, nu)
        while inner_it < max_iter and diff > max_error:
            inner_it += 1
            self.velocity_tentative_assemble()
            diff, errors = self.velocity_tentative_solve()
            assert (errors > 0).all()
            self.pressure_assemble(dt)
            error_p = self.pressure_solve(nu=nu)
            assert error_p > 0

        self.velocity_update(dt)

        # Propagate solutions u1->u2, u->u1
        for i in range(self._mesh.geometry.dim):
            self._u2[i].x.array[:] = self._u1[i].x.array
            self._u1[i].x.array[:] = self._u[i].x.array

        self._p.x.array[:] = self._ps.x.array[:]
        # Update ouptut u for post-processing
        self.u
        return diff

    @property
    def u(self):
        """
        Return the solution to the tentative velocity equation as a vector function
        """
        for ui, (Vi, map) in zip(self._u, self._Vi):
            self._sol_u.x.array[map] = ui.x.array
        return self._sol_u
