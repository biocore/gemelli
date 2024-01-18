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

DEFAULT_MTD = 0
DEFAULT_BL = None
DEFAULT_COMP = 3
DEFAULT_MSC = 0
DEFAULT_MFC = 0
DEFAULT_MFF = 0
DEFAULT_OPTSPACE_ITERATIONS = 5
DEFAULT_TENSALS_MAXITER = 25
DEFAULT_FMETA = None
DEFAULT_COND = None
DEFAULT_METACV = None
DEFAULT_COLCV = None
DEFAULT_TESTS = 10
DEFAULT_MATCH = True
DEFAULT_TRNSFRM = True
DEFAULT_TEMPTED_PC = 1
DEFAULT_TEMPTED_EP = 1e-4
DEFAULT_TEMPTED_SMTH = 1e-6
DEFAULT_TEMPTED_RES = 101
DEFAULT_TEMPTED_MAXITER = 20
DEFAULT_TEMPTED_RH = 'random'
DEFAULT_TEMPTED_RHC = 'random'
DEFAULT_TEMPTED_SVDC = True
DEFAULT_TEMPTED_SVDCN = 1
DESC_BIN = ("The feature table containing the "
            "samples over which metric should be computed.")
DESC_COUNTS = ("The feature table in biom format containing the "
               "samples over which metric should be computed.")
DESC_TREE = ("Phylogenetic tree containing tip identifiers that "
             "correspond to the feature identifiers in the table. "
             "This tree can contain tip ids that are not present "
             "in the table, but all feature ids in the table must "
             "be present in this tree.")
DESC_MINDEPTH = ("Minimum number of total number of "
                 "descendants (tips) to include a node. "
                 "Default value of zero will retain all nodes "
                 "(including tips).")
DESC_COMP = ("The underlying low-rank structure."
             " The input can be an integer "
             "(suggested: 1 < rank < 10) [minimum 2]."
             " Note: as the rank increases the runtime"
             " will increase dramatically.")
DESC_ITERATIONS = ("The number of iterations to optimize the solution"
                   " (suggested to be below 100; beware of overfitting)"
                   " [minimum 1]")
DESC_INIT = ("The number of initialization vectors. Larger values will"
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
DESC_SMETA = "Sample metadata file in QIIME2 formatting."
DESC_TAX_Q2 = ("Taxonomy file in QIIME2 formatting.")
DESC_TAX_SA = ("A tsv taxonomy file which contains a column labeled either "
               "'taxon' or 'taxonomy' (case insensitive) and the first column "
               "must be labeled "
               "'Feature ID'. A new taxonomy file will be created.")
DESC_T2T_TAX = ("The resulting tax2Tree taxonomy and will include taxonomy "
                "for both internal nodes and tips.")
DESC_STBL = ("A table with samples aggergated by subject. "
             "This can be used and input into Empress.")
DESC_SUBJ = ("Metadata column containing subject IDs to"
             " use for pairing samples. WARNING: if"
             " replicates exist for an individual ID at"
             " either state_1 to state_N, that subject will"
             " be mean grouped by default.")
DESC_COND = ("Metadata column containing state (e.g.,Time, BodySite)"
             " across which samples are paired."
             " At least one is required but up to four are allowed"
             " by other state inputs.")
QORD = ("A trajectory is an ordination that can be visualized "
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
QTREE = ("The input tree with all nodes matched in name to the features "
         "in the counts_by_node table.")
QTREECOUNT = ("A table with all tree internal nodes as features with the "
              "sum of all children of that node (i.e. Fast UniFrac).")
QBIPLOT = "A biplot of the (Robust Aitchison) RPCA feature loadings"
QADIST = "The Aitchison distance of the sample loadings from RPCA."
QACV = "The cross-validation reconstruction error."
QRCLR = "A rclr transformed table. Note: zero/missing values have NaNs"
DESC_METACV = ("Sample metadata file in QIIME2 formatting. "
               "Containing the columns with training and test labels.")
DESC_COLCV = ("Sample metadata column containing `train` and `test`"
              " labels to use for the cross-validation evaluation.")
DESC_TESTS = ("Number of random samples to choose for test split samples "
              "if sample metadata and a train-test column are not provided.")
DESC_TABLES = ("The collection of feature tables containing shared "
               "samples over which metric should be computed.")
DESC_TRAINTABLES = ("The tables to be projected on the first"
                    " principal components previously extracted"
                    " from a training set.")
DESC_TRAINTABLE = ("The table to be projected on the first"
                   " principal components previously extracted"
                   " from a training set.")
DESC_TRAINORDS = ("A joint-biplot of the (Robust Aitchison) RPCA"
                  " feature loadings produced from the training data.")
DESC_TRAINORD = ("A biplot of the (Robust Aitchison) RPCA"
                 " feature loadings produced from the training data.")
DESC_MATCH = ("Subsets the input tables to contain only features used in the"
              " training data. If set to False and the tables are not"
              " perfectly. Matched a ValueError will be produced.")
DESC_TRNSFRM = ("If set to false the function will expect `tables`"
                "to be dataframes already rclr transformed."
                " This is used for internal functionality in the "
                "joint-rpca function and is set to be only False.")
DESC_MTABLE = ("The biplot ordination in"
               " skbio.OrdinationResults format to match.")
DESC_MORD = ("The feature table in biom format containing the"
             "samples and features to match the ordination with.")
DESC_FM = "If set to True the features in the ordination will be matched."
DESC_SM = "If set to True the samples in the ordination will be matched."
DESC_MORDOUT = "A subset biplot with the input table."
DESC_CORRTBLORD = ("A joint-biplot or subset joint-biplot"
                   " of the (Robust Aitchison) RPCA feature loadings.")
DESC_CORRTBL = "A feature by feature correlation table."
DESC_TCOND = ("Metadata column containing time points"
              " across which samples are paired.")
DESC_REP = ('Choose how replicate samples are handled. If replicates are'
            'detected, "error" causes method to fail; "drop" will discard'
            ' all replicated samples; "random" chooses one representative at'
            ' random from among replicates.')
DESC_SVD = "Removes the mean structure of the temporal tensor."
DESC_SVDC = "Rank of approximation for average matrix in svd-centralize."
DESC_RES = ("Number of time points to evaluate the value"
            " of the temporal loading function.")
DESC_SMTH = ("Smoothing parameter for RKHS norm. Larger means "
             "smoother temporal loading functions.")
DESC_MXTR = "Maximum number of iteration in for rank-1 calculation."
DESC_EPS = ("Convergence criteria for difference between iterations "
            "for each rank-1 calculation.")
DESC_IO = ("Compositional biplot of subjects as points and"
           " features as arrows. Where the variation between"
           " subject groupings is explained by the log-ratio"
           " between opposing arrows.")
DESC_PIO = ("Compositional biplot of subjects as points from"
            " new data projected into a pre-generated space.")
DESC_SLO = ("Each components temporal loadings across the"
            "input resolution included as a column called"
            "'time_interval'.")
DESC_SVDO = ("The loadings from the SVD centralize"
             " function, used for projecting new data.")
DESC_TDIST = "Subject by subject distance matrix (not samples)."
DESC_PC = ("The pseudocount to add to the table before applying the "
           "transformation. Default is zero which will add the"
           " minimum non-zero value to all the values.")
DESC_TJNT = ("If flagged Joint-RPCA will not use the RCLR "
             "transformation and will instead assume that "
             "the data has already been transformed or "
             "normalized. Disabling the RCLR step will also "
             "disable any filtering steps. It will be expected "
             "that all filtering was done previously."
             "Default is to use RCLR.")
