# Copyright (C) 2022 Jørgen Schartum Dokken
#
# This file is part of Oasisx
# SPDX-License-Identifier:    MIT


from enum import Enum
from typing import Callable, List, Optional, Tuple, Union

from petsc4py import PETSc as _PETSc

import dolfinx.fem as _fem
import dolfinx.mesh as _dmesh
import numpy as np
import numpy.typing as npt
import ufl
from dolfinx import default_scalar_type

__all__ = ["DirichletBC", "PressureBC", "LocatorMethod"]


class LocatorMethod(Enum):
    """
    Search methods for Dirichlet BCs
    """

    GEOMETRICAL = 1
    TOPOLOGICAL = 2


LocatorMethod.TOPOLOGICAL.__doc__ = "Topogical search for dofs"
LocatorMethod.GEOMETRICAL.__doc__ = "Geometrical search for dofs"


class DirichletBC:
    """
    Create a Dirichlet boundary condition based on topological or geometrical info from the mesh
    This boundary condtion should only be used for velocity function spaces.

    Args:
        value: The value the degrees of freedom should have. It can be a float, a
            `dolfinx.fem.Constant` or a lambda-function.
        method: Locator method for marker.
        marker: If :py:obj:`oasisx.LocatorMethod.TOPOLOGICAL` the input should
            be a tuple of a mesh tag and the corresponding value for the entities to assign
            Dirichlet conditions to. If :py:obj:`oasisx.LocatorMethod.GEOMETRICAL` the input
            is a lambda function taking in x, y, and z coordinates (ordered as
            `[[x0, x1,...,xn], [y0,...,yn], [z0,...,zn]]`), and return a boolean marker where
            the ith entry indicates if `[xi, yi, zi]` should have a diriclet bc.

    Example:
        **Assigning a topological condition**

        .. highlight:: python
        .. code-block:: python

            entities = np.array([0,3,8],dtype=np.int32)
            values = np.full_like(entities, 2, dtype=np.int32)
            mt  = dolfinx.fem.meshtags(mesh, mesh.topology.dim-1, entities, values)
            bc = DirichletBC(5., LocatorMethod.TOPOLOGICAL, (mt, 2))

        **Assigning a geometrical condition**

        .. highlight:: python
        .. code-block:: python

            bc = DirchletBC(dolfinx.fem.Constant(mesh, 3.), lambda x: np.isclose(x[0]))
    """

    _method: LocatorMethod
    _entities: npt.NDArray[np.int32]  # List of entities local to process
    _e_dim: int  # Dimension of entities

    _locator: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.bool_]]
    _dofs: npt.NDArray[np.int32]
    _value: Union[
        np.float64,
        _fem.Constant,
        Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    ]
    _bc: _fem.DirichletBC
    _u: Optional[_fem.Function]

    __slots__ = tuple(__annotations__)

    def __init__(
        self,
        value: Union[
            np.float64,
            _fem.Constant,
            Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
        ],
        method: LocatorMethod,
        marker: Union[
            Tuple[_dmesh.MeshTags, np.int32],
            Callable[[npt.NDArray[np.float64]], npt.NDArray[np.bool_]],
        ],
    ):
        if method == LocatorMethod.GEOMETRICAL:
            self._method = method
            setattr(self, "_locator", marker)
        elif method == LocatorMethod.TOPOLOGICAL:
            self._method = method
            self._entities = marker[0].find(marker[1])  # type: ignore
            self._e_dim = marker[0].dim  # type:ignore
        self._value = value

    def set_dofs(self, dofs: npt.NDArray[np.int32]):
        self._dofs = dofs

    def _locate_dofs(self, V: _fem.FunctionSpace):
        """
        Locate all dofs satisfying the criterion
        """
        if self._method == LocatorMethod.GEOMETRICAL:
            self._dofs = _fem.locate_dofs_geometrical(V, self._locator)  # type:ignore
        elif self._method == LocatorMethod.TOPOLOGICAL:
            V.mesh.topology.create_connectivity(self._e_dim, V.mesh.topology.dim)
            self._dofs = _fem.locate_dofs_topological(V, self._e_dim, self._entities)

    def create_bc(self, V: _fem.FunctionSpace):
        """ """
        if not hasattr(self, "_dofs"):
            self._locate_dofs(V)

        try:
            self._bc = _fem.dirichletbc(self._value, self._dofs, V)  # type:ignore
        except AttributeError:
            self._u = _fem.Function(V)
            self._u.interpolate(self._value)  # type:ignore
            self._bc = _fem.dirichletbc(self._u, self._dofs)

    def update_bc(self):
        """
        Update the underlying function if input value is a lambda function
        """
        if hasattr(self, "_u"):
            self._u.interpolate(self._value)

    def apply(self, x: _PETSc.Vec):  # type: ignore
        """
        Apply boundary condition to a PETSc vector
        """
        _fem.petsc.set_bc(x, [self._bc])


class PressureBC:
    """
    Create a Pressure boundary condition (natural bc) based on a set of facets of a
    mesh and some value.

    Args:
        value: The value the degrees of freedom should have. It can be a float, a
            `dolfinx.fem.Constant` or a lambda-function. If `value` is a lambda-function it
            is interpolated into the pressure space.
        marker:  Tuple of a mesh tag and the corresponding value for the entities to assign
            Dirichlet conditions to. The meshtag dimension has to be `mesh.topology.dim -1`.


    Example:
        **Assigning a constant condition**

        .. highlight:: python
        .. code-block:: python

            entities = np.array([0,3,8],dtype=np.int32)
            values = np.full_like(entities, 2, dtype=np.int32)
            mt  = dolfinx.fem.meshtags(mesh, mesh.topology.dim-1, entities, values)
            bc = DirichletBC(5., (mt, 2))

        **Assigning a time-dependent condition**

        .. highlight:: python
        .. code-block:: python

            class OutletPressure():
                def __init__(self, t:float):
                    self.t = t
                def eval(self, x: numpy.typing.NDArray[np.float64]):
                    return self.t*x[0]

            p = OutletPressure(0.)
            entities = np.array([0,3,8],dtype=np.int32)
            values = np.full_like(entities, 2, dtype=np.int32)
            mt  = dolfinx.fem.meshtags(mesh, mesh.topology.dim-1, entities, values)
            bc = PressureBC(p,  (mt, 2))

            Q = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
            # Create appropriate structures for assigning bcs
            bc.create_bc(Q)
            # Update time in bc
            p.t = 1
    """

    _subdomain_data: _dmesh.MeshTags
    _subdomain_id: Union[np.int32, Tuple[np.int32]]
    _value: Union[
        np.float64,
        _fem.Constant,
        Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    ]
    _u: _fem.Function
    _rhs: List[ufl.form.Form]
    _bc: _fem.DirichletBC
    __slots__ = tuple(__annotations__)

    def __init__(
        self,
        value: Union[
            np.float64,
            _fem.Constant,
            Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
        ],
        marker: Tuple[_dmesh.MeshTags, Union[np.int32, Tuple[np.int32]]],
    ):
        self._subdomain_data, self._subdomain_id = marker
        self._value = value

    def create_bcs(self, V: _fem.FunctionSpace, Q: _fem.FunctionSpace):
        """
        Create boundary conditions for the pressure conditions for a given velocity and
        pressure function space

        Args:
            V: The velocity function space
            Q: The pressure function space
        """
        mesh = V.mesh
        assert mesh.topology == self._subdomain_data.topology
        # Create pressure "Neumann" condition
        v = ufl.TestFunction(V)
        ds = ufl.Measure(
            "ds",
            domain=mesh,
            subdomain_data=self._subdomain_data,
            subdomain_id=self._subdomain_id,
        )
        n = ufl.FacetNormal(mesh)
        try:
            rhs = [self._value * n_i * v.dx(i) * ds for i, n_i in enumerate(n)]
        except TypeError:
            # If input is lambda function interpolate into local function
            self._u = _fem.Function(Q)
            self._u.interpolate(self._value)  # type: ignore
            rhs = [self._u * n_i * v.dx(i) * ds for i, n_i in enumerate(n)]

        # Create rhs contribution from natural boundary condition
        self._rhs = rhs

        # Create homogenuous boundary condition for pressure correction eq
        boundary_facets = self._subdomain_data.find(self._subdomain_id)  # type: ignore
        mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
        dofs = _fem.locate_dofs_topological(Q, mesh.topology.dim - 1, boundary_facets)
        self._bc = _fem.dirichletbc(default_scalar_type(0.0), dofs, Q)

    def update_bc(self):
        """
        Update boundary condition if input-value is a lambda function
        """
        if hasattr(self, "_u"):
            self._u.interpolate(self._value)

    @property
    def bc(self) -> _fem.DirichletBC:
        return self._bc

    def rhs(self, i: int) -> _fem.Form:
        assert i < len(self._rhs)
        return self._rhs[i]
