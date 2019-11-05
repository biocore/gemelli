v0.0.5 (2019-06-12)

### Features
* The output of the standalone and QIIME2 -CLI now return a state ordination
    * Note that only the standalone can handle more than one state due to the lack of lists in QIIME2

### Bug fixes

* Fixes in `_transformer.py` and `ctf.py` to utalize QIIME2 metadata passing, this fixes the dtype bool dropping issues in metadata
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


(2019-05-17)

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

Original "working" code

### Features

### Backward-incompatible changes [stable]

### Backward-incompatible changes [experimental]

### Performance enhancements

### Bug fixes

### Deprecated functionality [stable]

### Deprecated functionality [experimental]

### Miscellaneous
