# + [markdown]
# # An introduction to the light format
# Author: Jørgen S. Dokken
#
# SPDX-License-Identifier:    MIT
# -

# + [markdown]
# We start a new cell explicitly by using the "+" sign and can explicitly end
# it by using the "-" sign.
# -

# If we do not use explicit indicators for starting or ending a cell, it creates a
# new cell whenever there is a blank line

# This is a new cell

# Comments can be added to code by not adding a new line before the code
import mypackage

# Next we define two numbers, `a` and `b`

a = 1
b = 3

# and add them together

c = mypackage.addition(a, b)

# We check the result

assert c == a + b
