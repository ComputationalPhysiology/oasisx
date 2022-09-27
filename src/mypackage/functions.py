

# Copyright (C) 2022 Jørgen Schartum Dokken
#
# This file is part of mypackage
# SPDX-License-Identifier:    MIT

__all__ = ["addition", "print_add"]


def addition(a: int, b: int) -> int:
    """ Computes a+b """
    return a + b


def print_add(a: int, b: int) -> int:
    """ Computes and prints a + b"""
    c = a + b
    print(c)
    return c
