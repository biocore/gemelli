# Configuration file where you can set the parameter default values and
# descriptions.
DEFAULT_COMP = 3
DEFAULT_MSC = 0
DEFAULT_MFC = 0
DEFAULT_MAXITER = 25
DEFAULT_FMETA = None
DEFAULT_COND = None

DESC_INIT = ("The number of initialization vectors. Larger values will"
             "give more accurate factorization but will be more "
             "computationally expensive [minimum 1]")
DESC_ITERATIONSALS = ("Max number of Alternating Least Square (ALS)"
                      " optimization iterations (suggested to be below 100;"
                      " beware of overfitting) [minimum 1]")
DESC_ITERATIONSRTPM = ("Max number of Robust Tensor Power Method (RTPM)"
                       " optimization iterations (suggested to be below 100;"
                       " beware of overfitting) [minimum 1]")
DESC_COMP = ("The underlying low-rank structure (suggested: 2 < rank < 10)"
             " [minimum 2]")
DESC_MSC = "Minimum sum cutoff of sample across all features"
DESC_MFC = "Minimum sum cutoff of features across all samples"
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
QORD = ("A trajectory is an ordination that can be visualized"
        "over time or another context.")
QDIST = ("A sample-sample distance matrix generated from "
         " the euclidean distance of the subject-state "
         "ordinations and itself.")
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
