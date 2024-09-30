
v0.0.12 (2024-10-01)

### Bug fixes

* Joint-RPCA re-centering bug.
    * see issue #97
* Updated for the newest version of Pandas.
    * see issue #96
* Reduce unecessary numpy warnings regarding log on unseen data.
    * see issue #78
* Fix filter and ordering with TEMPTED
    * see issue #92 and issue #93

v0.0.11 (2024-06-24)

### Bug fixes

* Passes non-centered tables if non-centered flag is passed.
    * see issue #82
* setup.py packages html files so visualization commands works from pip install.
    * see issue #83
    * see issue #87
* Passes no filtering to the QIIME2/API TEMPTED command since filtering and transformation should already be done.
    * see issue #90

v0.0.10 (2023-12-01)

### Bug fixes

* Removed np.int types
    * fixes bug that prevents use in newer versions of numpy / QIIME2
        * see issue #71

### Features [experimental]

* rpca projection of new data and with cross validation
    * A new function for RPCA where new data can be projected into an existing ordination.
    * Also allows for internal CV, where the hold out data is projected into the training ordination, which is then used to re-build the test data, and the error between the projection and the real test data is calculated.
        * see issue #70
* tempted.py and assocated tests/commands/tutorials
    * Added in a python implementation TEMPTED, more details on the methods inm this paper [here](https://www.biorxiv.org/content/10.1101/2023.07.26.550749v1).
* joint-rpca and associated tests/commands/tutorials
    * Joint-RPCA is an extention of RPCA for multiple omics types.
* qc_rarefaction
    * This function compares the mantel correlation of distances to the abs. differences in sample sum between rarefied and unrarefied input data. This is an easy check to ensure the results are not bieng significantly altered by non-rarefaction is cases of large differences (e.g., low-biomass) or where the sample sum differences can not be seperated from the phenotypes/low-rank structure of the data (i.e., deep sequencing of controls and shallow of sick).
        * see issue #70

### Deprecated functionality

* auto_rpca across the package and rank_estimate in optspace.py
    * was not performing correctly and no reasonable quick fix is available.
        * see issue #70
    * **Change from `auto_rpca` to `gemelli.rpca import rpca`**

v0.0.9 (2023-05-24)

### Bug fixes

* Fixes in `preprocessing.py` function
    * fixes bug that can drastically change taxonomic assigmennt
        * see issue #66

v0.0.8 (2022-03-28)

### Bug fixes

* Fixes in `rpca.py` function
    * fixes bug that passes QIIME2 Metadata instead of the expected pandas dataframe
        * see issue #57

v0.0.7 (2022-01-19)

### Features [stable]

* Moved CI from Travis to GitHub actions. 
* Updated existing tutorials based on comments. (see https://forum.qiime2.org/t/ctf-linear-mixed-model/20622).
* Added commands output just the rclr transformed table. 

### Features [experimental]

* Phylogenetic RCLR and integration into RPCA/CTF
    * added phylogenetic weighting in rclr in `preprocessing.py`
    * added commands in `scripts` and `q2` to perform these features with and without taxonomy
    * added tutorials in `ipynb/tutorials` to demonstrate the commands/workflows

### Bug fixes

* Fixes in `optspace.py` function `rank_estimate` and  `factorization.py` function `tenals`
    * fixes two subject/state tensor
        * see https://forum.qiime2.org/t/gemelli-argmin-error/21796 and issue #38
    * added test in `test_method.py` for two subject or state tensor 
* Fixes in `preprocessing.py`
    * fixes when the ordering/structure of the _tensor_ could impact the rlcr results by sorting to ensure reproducibility.
        * see https://forum.qiime2.org/t/gemelli-argmin-error/21796 and thanks to Xinhe Qi for reporting. 
    * added tests to ensure sorting of samples/features does not impact results.

v0.0.6 (2020-09-27)

### Features

* Robust Aitchison PCA
    * with manually chosen n_components
    * with auto chosen n_components
    * associated tests added for function, standalone, & QIIME2
* Robust centered log-ratio (rclr) transformation
    * allows users to use the transform alone outside CTF/RPCA
    * associated tests added and extra tests for rclr added  
* Tutorials
    * Cross-sectional for RPCA (standalone & QIIME2)
    * Repeated measure for CTF (standalone & QIIME2)


### Miscellaneous

* auto_rpca rank estimation min rank now 3 
* added citations for CTF & RPCA (linked when appropriate)
* updated the function descriptions

v0.0.5 (2019-06-12)

### Features

* The output of the standalone and QIIME2 -CLI now return a state ordination
    * Note that only the standalone can handle more than one state due to the lack of lists in QIIME2

### Bug fixes

* Fixes in `_transformer.py` and `ctf.py` to utilize QIIME2 metadata passing, this fixes the dtype bool dropping issues in metadata
* Removed imports of QIIME2 metadata in `ctf.py`, this will allow the standalone commandline to run without having QIIME2 installed.
* Update pandas future error of .loc[new_index, :] to .reindex(new_index)
* Centered the subject biplot
* Added flake8 testing to travis

v0.0.4 (2019-06-12)

### Features

* Tensor Building and RCLR transformation in `preprocessing.rclr` and `preprocessing.build`
    * N-mode tensor building and transformation
    * Mean of counts for subject-conditional pairs with several samples

### Backward-incompatible changes [stable]

* In `preprocessing.build`:
    * pervious -> current
    * build().sample_order -> build().subject_order
    * build().temporal_order -> build().condition_orders
        * as a list for N possible condition(s)
    * build().tensor -> build().counts

### Backward-incompatible changes [experimental]

### Performance enhancements

* tensor building and transformation

### Bug fixes

* line 369-360 in `factorization.tenals` causes np.nan(s) in solution
    * fixed by added pseudocount if any nan in solution

* line 178-179 in `factorization.TenAls` 
    * was previously checking if all missing/zero not if there were no missing/zero as intended

### Deprecated functionality [stable]

* In `preprocessing.rclr` and `preprocessing.build`:
    * build().transform() -> `preprocessing.rclr` as standalone function

### Deprecated functionality [experimental]

### Miscellaneous

* line 175 in `factorization.TenAls` to send ValueError if input is not numpy array


v0.0.3 (2019-05-17)

### Features

* Tensor factorization in `factorization.tenals` and `factorization.TenALS`
    * Accomplished by turning hard-coded operations into for loops and generalizing vectorization using tensor contractions

* Khatri Rao Product

### Backward-incompatible changes [stable]

### Backward-incompatible changes [experimental]

* `factorization.tenals` returns a list of matrices `loadings` instead of `u1, u2, u3` tuple, along with other arguments

### Performance enhancements

* tensor contraction

### Bug fixes

### Deprecated functionality [stable]

### Deprecated functionality [experimental]

### Miscellaneous

* various type and spacing fixes

Original "working" code (v0.0.1-2)

### Features

### Backward-incompatible changes [stable]

### Backward-incompatible changes [experimental]

### Performance enhancements

### Bug fixes

### Deprecated functionality [stable]

### Deprecated functionality [experimental]

### Miscellaneous
