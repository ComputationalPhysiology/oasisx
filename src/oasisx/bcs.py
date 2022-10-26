# Copyright (C) 2022 Jørgen Schartum Dokken
#
# This file is part of Oasisx
# SPDX-License-Identifier:    MIT


from enum import Enum
from typing import Callable, Union, Tuple

import dolfinx.mesh as _dmesh
import dolfinx.fem as _fem
import numpy as np
import numpy.typing as npt

__all__ = ["DirichletBC", "LocatorMethod"]


class LocatorMethod():
    TOPOLOGICAL = 1
    GEOMETRICAL = 2


class DirichletBC():
    """


    """
    _method: LocatorMethod
    _entities: npt.NDArray[np.int32]  # List of entities local to process
    _e_dim: int  # Dimension of entities

    _locator: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.bool_]]
    _dofs: npt.NDArray[np.int32]
    _value: Union[np.float64, _fem.Constant, Callable[[
        npt.NDArray[np.float64]], npt.NDArray[np.float64]]]
    _bc: _fem.DirichletBCMetaClass
    _u: _fem.Function

    __slots__ = tuple(__annotations__)

    def __init__(self, value: Union[np.float64, _fem.Constant, Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]],
                 topological: Tuple[_dmesh.MeshTagsMetaClass, np.int32] = None,
                 geometrical: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.bool_]] = None):
        if topological is None and geometrical is None:
            raise RuntimeError("Need to supply locators for the mesh topology or the mesh geometry")
        elif topological is not None and geometrical is not None:
            raise RuntimeError("Cannot use topogical and geometrical locators at the same time")
        elif geometrical is not None:
            self._method = LocatorMethod.GEOMETRICAL
            self._locator = geometrical
        elif topological is not None:
            self._method = LocatorMethod.TOPOLOGICAL
            self._entities = topological[0].find(topological[1])
            self._e_dim = topological[0].dim
        self._dofs = None
        self._value = value

    def set_dofs(dofs: npt.NDArray[np.int32]):
        self._dofs = dofs

    def _locate_dofs(self, V: _fem.FunctionSpace):
        """
        Locate all dofs satisfying the criterion
        """
        if self._method == LocatorMethod.GEOMETRICAL:
            self._dofs = _fem.locate_dofs_geometrical(V, self._locator)
        elif self._method == LocatorMethod.TOPOLOGICAL:
            self._dofs = _fem.locate_dofs_topological(V, self._e_dim, self._entities)

    def create_bc(self, V: _fem.FunctionSpace):
        """

        """
        if self._dofs is None:
            self._locate_dofs(V)

        try:
            self._bc = _fem.dirichletbc(self._value, self._dofs, V)
        except AttributeError:
            self._u = _fem.Function(V)
            self._u.interpolate(self._value)
            self._bc = _fem.dirichletbc(self._u, self._dofs)
