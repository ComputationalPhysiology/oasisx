
# Copyright (C) 2022 Jørgen Schartum Dokken
#
# This file is part of my_package
# SPDX-License-Identifier:    MIT

import importlib.metadata

from .functions import addition, print_add

__version__ = importlib.metadata.version(__package__)


__all__ = ["addition", "print_add"]
