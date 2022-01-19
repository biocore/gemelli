[![Build Status](https://travis-ci.org/biocore/gemelli.svg?branch=master)](https://travis-ci.org/biocore/gemelli)

# Gemelli

Gemelli is a tool box for running both Robust Aitchison PCA (RPCA) and Compositional Tensor Factorization (CTF) on _sparse_ compositional omics datasets.

RPCA can be used on cross-sectional datasets where each subject is sampled only once. CTF can be used on repeated-measure data where each subject is sampled multiple times (e.g. longitudinal sampling). Both methods are [_unsupervised_](https://en.wikipedia.org/wiki/Unsupervised_learning) and aim to describe sample/subject variation and the biological features that separate them. 

The preprocessing transform for both RPCA and CTF is the robust centered log-ratio transform (rlcr) which accounts for sparse data (i.e. many missing/zero values). Details on the rclr can be found [here](https://msystems.asm.org/content/4/1/e00016-19) and a interactive introduction into the transformation can be found [here](https://github.com/biocore/gemelli/blob/master/ipynb/tutorials/introduction.ipynb). In short, the rclr log transforms the observed (nonzero) values before centering. RPCA and CTF then perform a matrix or tensor factorization on only the observed values after rclr transformation, similar to [Aitchison PCA](https://academic.oup.com/biomet/article-abstract/70/1/57/240898?redirectedFrom=fulltext) performed on dense data. If the data also has an associated phylogeny it can be incorporated through the phylogenetic rclr, details can be found [here](https://github.com/biocore/gemelli/blob/master/ipynb/tutorials/Phylogenetic-RPCA-moving-pictures.ipynb).

# Installation

To install the most up to date version of gemelli, run the following command

    # pip (only supported for QIIME2 >= 2018.8)
    pip install gemelli

**Note**: that gemelli is not compatible with python 2, and is compatible with Python 3.4 or later. 

# Documentation

Gemelli can be run standalone or through [QIIME2](https://qiime2.org/) and as a python API or CLI. 

## Cross-sectional study (i.e. one sample per subject) with RPCA

If you have a [cross-sectional study design](https://en.wikipedia.org/wiki/Cross-sectional_study) with only one sample per subject then RPCA is the appropriate method to use in gemelli.  There are two commands within RPCA. The first is `rpca` and the second is `auto-rpca`. The only difference is that `auto-rpca` automatically estimates the underlying-rank of the matrix and requires no input for the `n_components` parameter. In the `rpca` command the `n_components` must be set explicitly. For examples of using RPCA we provide tutorials below exploring the microbiome between body sites.

### Tutorials

#### Tutorials with QIIME2

* [RPCA QIIME2 CLI](https://github.com/biocore/gemelli/blob/master/ipynb/tutorials/RPCA-moving-pictures.ipynb)
* [Phylogenetic RPCA QIIME2 CLI](https://github.com/biocore/gemelli/blob/master/ipynb/tutorials/Phylogenetic-RPCA-moving-pictures.ipynb)

#### Standalone tutorial outside of QIIME2

* [RPCA Python API & CLI](https://github.com/biocore/gemelli/blob/master/ipynb/tutorials/RPCA-moving-pictures-standalone-cli-and-api.ipynb)

## Repeated measures study (i.e. one sample per subject) with CTF

### Tutorials

If you have a [repeated measures study design](https://en.wikipedia.org/wiki/Repeated_measures_design) with multiple samples per subject over time or space then CTF is the appropriate method to use in gemelli. For optimal results CTF requires samples for each subject in each time or space measurement. In some cases, this can require binning time my larger windows (e.g. instead of days use months). For examples of using CTF we provide a microbiome time series IBD study in the tutorials below.

#### Tutorials with QIIME2

* [CTF QIIME2 CLI](https://github.com/biocore/gemelli/blob/master/ipynb/tutorials/IBD-Tutorial-QIIME2-CLI.md)
* [CTF QIIME2 API](https://github.com/biocore/gemelli/blob/master/ipynb/tutorials/IBD-Tutorial-QIIME2-API.ipynb)
* [Phylogenetic CTF QIIME2 CLI](https://github.com/biocore/gemelli/blob/master/ipynb/tutorials/Phylogenetic-IBD-Tutorial-QIIME2-CLI.ipynb)
* [Phylogenetic CTF QIIME2 API](https://github.com/biocore/gemelli/blob/master/ipynb/tutorials/Phylogenetic-IBD-Tutorial-QIIME2-API.ipynb)

#### Standalone tutorial outside of QIIME2

* [CTF Standalone Python API](https://github.com/biocore/gemelli/blob/master/ipynb/tutorials/IBD-Tutorial-standalone-API.ipynb)

# Citations

If you found this tool useful please cite the method(s) you used:

## Citation for CTF

```
Martino, C. and Shenhav, L. et al. Context-aware dimensionality reduction deconvolutes gut microbial community dynamics. Nat. Biotechnol. (2020) doi:10.1038/s41587-020-0660-7
```

```
@article {Martino2020,
	author = {Martino, Cameron and Shenhav, Liat and Marotz, Clarisse A and Armstrong, George and McDonald, Daniel and V{\'a}zquez-Baeza, Yoshiki and Morton, James T and Jiang, Lingjing and Dominguez-Bello, Maria Gloria and Swafford, Austin D and Halperin, Eran and Knight, Rob},
	title = {Context-aware dimensionality reduction deconvolutes gut microbial community dynamics},
	year = {2020},
	journal = {Nature biotechnology},
}
```


## Citation for RPCA

```
Martino, C. et al. A Novel Sparse Compositional Technique Reveals Microbial Perturbations. mSystems 4, (2019)
```

```
@article {Martino2019,
	author = {Martino, Cameron and Morton, James T. and Marotz, Clarisse A. and Thompson, Luke R. and Tripathi, Anupriya and Knight, Rob and Zengler, Karsten},
	editor = {Neufeld, Josh D.},
	title = {A Novel Sparse Compositional Technique Reveals Microbial Perturbations},
	volume = {4},
	number = {1},
	elocation-id = {e00016-19},
	year = {2019},
	doi = {10.1128/mSystems.00016-19},
	publisher = {American Society for Microbiology Journals},
	URL = {https://msystems.asm.org/content/4/1/e00016-19},
	eprint = {https://msystems.asm.org/content/4/1/e00016-19.full.pdf},
	journal = {mSystems}
}
```

## Other Resources

- The compositional data [wiki](https://en.wikipedia.org/wiki/Compositional_data)
- The code for OptSpace was translated to python from a [MATLAB package](http://swoh.web.engr.illinois.edu/software/optspace/code.html) maintained by Sewoong Oh (UIUC).
- [TenAls translated from Sewoong Oh](http://swoh.web.engr.illinois.edu/software/optspace/code.html)
- Transforms and PCoA : [Scikit-bio](http://scikit-bio.org)
- Data For Examples : [Qiita](https://qiita.ucsd.edu/)
