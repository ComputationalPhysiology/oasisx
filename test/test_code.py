# Copyright (C) 2022 Jørgen Schartum Dokken
#
# This file is part of my_package
# SPDX-License-Identifier:    MIT

from mypackage import addition


def test_addition():
    a = 5
    b = 3
    assert addition(a, b) == a + b
