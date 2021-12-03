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

DEFAULT_COMP = 3
DEFAULT_MSC = 0
DEFAULT_MFC = 0
DEFAULT_MFF = 0
DEFAULT_OPTSPACE_ITERATIONS = 5
DEFAULT_TENSALS_MAXITER = 25
DEFAULT_FMETA = None
DEFAULT_COND = None
DESC_COMP = ("The underlying low-rank structure."
             " The input can be an integer "
             "(suggested: 1 < rank < 10) [minimum 2]."
             " Note: as the rank increases the runtime"
             " will increase dramatically.")
DESC_ITERATIONS = ("The number of iterations to optimize the solution"
                   " (suggested to be below 100; beware of overfitting)"
                   " [minimum 1]")
DESC_INIT = ("The number of initialization vectors. Larger values will "
             "give more accurate factorization but will be more "
             "computationally expensive [minimum 1]")
DESC_ITERATIONSALS = ("Max number of Alternating Least Square (ALS)"
                      " optimization iterations (suggested to be below 100;"
                      " beware of overfitting) [minimum 1]")
DESC_ITERATIONSRTPM = ("Max number of Robust Tensor Power Method (RTPM)"
                       " optimization iterations (suggested to be below 100;"
                       " beware of overfitting) [minimum 1]")
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
DESC_OUT = "Location of output files."
DESC_FMETA = "Feature metadata file in QIIME2 formatting."
DESC_BIN = "Input table in biom format."
DESC_SMETA = "Sample metadata file in QIIME2 formatting."
DESC_SUBJ = ("Metadata column containing subject IDs to"
             " use for pairing samples. WARNING: if"
             " replicates exist for an individual ID at"
             " either state_1 to state_N, that subject will"
             " be mean grouped by default.")
DESC_COND = ("Metadata column containing state (e.g.,Time, BodySite)"
             " across which samples are paired."
             " At least one is required but up to four are allowed"
             " by other state inputs.")
QORD = ("A '%s trajectory ordination' that can be visualized "
        "over time or another context.")
QDIST = ("A sample-sample distance matrix generated from "
         " the Euclidean distance of the subject-state "
         "ordinations and itself. Similar to the distance matrix produced by"
         " RPCA except it also accounts for the subject and state context.")
QLOAD = ("Compositional biplot of subjects as points and features as arrows."
         " Where the variation between subject groupings is explained by the"
         " log-ratio between opposing arrows. "
         "WARNING: The % variance explained is only spread over n_components "
         "and can be inflated.")
QSOAD = ("Compositional biplot of states as points and features as arrows."
         " Where the variation between subject groupings is explained by the"
         " log-ratio between opposing arrows. "
         "WARNING: The % variance explained is only spread over n_components "
         "and can be inflated.")
