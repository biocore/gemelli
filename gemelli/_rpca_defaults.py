# ----------------------------------------------------------------------------
# Copyright (c) 2019--, gemelli development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

# Configuration file where you can set the parameter default values and
# descriptions. This is used by both the standalone RPCA and QIIME 2 RPCA sides
# of gemelli.

DEFAULT_RANK = 3
DEFAULT_MSC = 500
DEFAULT_MFC = 10
DEFAULT_MFF = 0
DEFAULT_ITERATIONS = 5

DESC_RANK = ("The underlying low-rank structure."
             " The input can be an integer "
             "(suggested: 1 < rank < 10) [minimum 2]."
             " Note: as the rank increases the runtime"
             " will increase dramatically.")
DESC_MSC = ("Minimum sum cutoff of sample across all features. "
            "The value can be at minimum zero and must be an whole"
            " integer. It is suggested to be greater than or equal"
            " to 500.")
DESC_MFC = ("Minimum sum cutoff of features across all samples. "
            "The value can be at minimum zero and must be an whole"
            " integer")
DESC_MFF = ("Minimum percentage of samples a feature must appear"
            " with a value greater than zero. This value can range"
            " from 0 to 100 with decimal values allowed.")
DESC_ITERATIONS = ("The number of iterations to optimize the solution"
                   " (suggested to be below 100; beware of overfitting)"
                   " [minimum 1]")
