# Copyright (C) 2022 Jørgen Schartum Dokken
#
# This file is part of Oasisx
# SPDX-License-Identifier:    MIT


from enum import Enum
from typing import Callable, Optional, Tuple, Union

import dolfinx.fem as _fem
import dolfinx.mesh as _dmesh
import numpy as np
import numpy.typing as npt
from petsc4py import PETSc as _PETSc

__all__ = ["DirichletBC", "LocatorMethod"]


class LocatorMethod(Enum):
    """
    Search methods for Dirichlet BCs
    """
    TOPOLOGICAL = 1  # doc: ""Topological search"""
    GEOMETRICAL = 2  # doc: """Geometrical search"""


class DirichletBC():
    """
    Create a Dirichlet boundary condition based on topological or geometrical info from the mesh

    Args:
        value: The value the degrees of freedom should have. It can be a float, a
            `dolfinx.fem.Constant` or a lambda-function.
        method: `oasisx.LocatorMethod`.
        marker: If :py:obj:`oasisx.LocatorMethod.TOPOLOGICAL` the input in `marker` should
            be a tuple of a mesh tag and the corresponding value for the entities to assign
            Dirichlet conditions to. If :py:obj:`oasisx.LocatorMethod.GEOMETRICAL` the marker
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
    _value: Union[np.float64, _fem.Constant, Callable[[
        npt.NDArray[np.float64]], npt.NDArray[np.float64]]]
    _bc: _fem.DirichletBCMetaClass
    _u: Optional[_fem.Function]

    __slots__ = tuple(__annotations__)

    def __init__(
        self, value: Union[np.float64, _fem.Constant,
                           Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]],
        method: LocatorMethod,
        marker: Union[Tuple[_dmesh.MeshTagsMetaClass, np.int32],
                      Callable[[npt.NDArray[np.float64]], npt.NDArray[np.bool_]]]
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
            self._dofs = _fem.locate_dofs_topological(V, self._e_dim, self._entities)

    def create_bc(self, V: _fem.FunctionSpace):
        """

        """
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
        if self._u is not None:
            self._u.interpolate(self._value)

    def apply(self, x: _PETSc.Vec):
        """
        Apply boundary condition to a PETSc vector
        """
        _fem.petsc.set_bc(x, [self._bc])
