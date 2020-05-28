[![Build Status](https://travis-ci.org/cameronmartino/gemelli.svg?branch=master)](https://travis-ci.org/cameronmartino/gemelli)
[![Coverage Status](https://coveralls.io/repos/github/cameronmartino/gemelli/badge.svg?branch=master)](https://coveralls.io/github/cameronmartino/gemelli?branch=master)

**gemelli is still being developed, so backwards-incompatible changes might occur.**
**If you have any questions, feel free to contact the development team at cmartino at eng.ucsd.edu.**

Gemelli is a tool box for running tensor factorization on sparse compositional omics datasets. Gemelli performs unsupervised dimensionality reduction of spatiotemporal microbiome data. The outlut of gemelli helps to resolve spatiotemporal subject variation and the biological features that separate them. 

# Installation

To install the most up to date version of deicode, run the following command

    # pip (only supported for QIIME2 >= 2018.8)
    pip install gemelli

**Note**: that gemelli is not compatible with python 2, and is compatible with Python 3.4 or later. 

# Documentation

## Tutorials 

There are several ways to run CTF through gemelli. Here we provide a tutorial of a time series IBD data set in each of the different formats:

### Command Line Interface
* [QIIME2 CLI](https://github.com/cameronmartino/gemelli/blob/master/ipynb/tutorials/IBD-Tutorial-QIIME2-CLI.md)
* [Standalone Python CLI](https://github.com/cameronmartino/gemelli/blob/master/ipynb/tutorials/IBD-Tutorial-standalone-CLI.md)

### Python API
* [QIIME2 API](https://github.com/cameronmartino/gemelli/blob/master/ipynb/tutorials/IBD-Tutorial-QIIME2-API.ipynb)
* [Standalone Python API](https://github.com/cameronmartino/gemelli/blob/master/ipynb/tutorials/IBD-Tutorial-standalone-API.ipynb)

## Other Resources

Named after gemelli by alighiero boetti and also the pasta. 

[TenAls translated from Sewoong Oh](http://swoh.web.engr.illinois.edu/software/optspace/code.html)
