[![Build Status](https://travis-ci.org/cameronmartino/gemelli.svg?branch=master)](https://travis-ci.org/cameronmartino/gemelli)
[![Coverage Status](https://coveralls.io/repos/github/cameronmartino/gemelli/badge.svg?branch=master)](https://coveralls.io/github/cameronmartino/gemelli?branch=master)

**gemelli is still being developed, so backwards-incompatible changes might occur.**
**If you have any questions, feel free to contact the development team at cmartino at eng.ucsd.edu.**

# gemelli

Gemelli is a tool box for running tensor factorization on sparse compositional omics datasets. Gemelli performs unsupervised dimensionality reduction of spatiotemporal microbiome data. The outlut of gemelli helps to resolve spatiotemporal subject variation and the biological features that separate them. 

## Installation

To install the most up to date version of deicode, run the following command

    # pip (only supported for QIIME2 >= 2018.8)
    pip install gemelli

**Note**: that gemelli is not compatible with python 2, and is compatible with Python 3.4 or later. 

## Using gemelli  inside [QIIME 2](https://qiime2.org/)

A QIIME2 tutorial can be found [here](https://github.com/cameronmartino/gemelli/blob/master/ipynb/tutorials/QIIME2-jansson-ibd-tutorial.md).

`Note: a more formal tutorial is coming soon.`

```bash
$ qiime gemelli ctf --help

Usage: qiime gemelli ctf [OPTIONS]

  Gemelli resolves spatiotemporal subject variation and the biological
  features that separate them. In this case, a subject may have several
  paired samples, where each sample may be a time point. The output is akin
  to conventional beta-diversity analyses but with the paired component
  integrated in the dimensionality reduction.

Inputs:
  --i-table ARTIFACT FeatureTable[Frequency]
                         Input table in biom format.                [required]
Parameters:
  --m-sample-metadata-file METADATA...
    (multiple            Sample metadata file in QIIME2 formatting.
     arguments will be   
     merged)                                                        [required]
  --p-individual-id-column TEXT
                         Metadata column containing subject IDs to use for
                         pairing samples. WARNING: if replicates exist for an
                         individual ID at either state_1 to state_N, that
                         subject will be mean grouped by default.   [required]
  --p-state-column TEXT  Metadata column containing state (e.g.,Time,
                         BodySite) across which samples are paired. At least
                         one is required but up to four are allowed by other
                         state inputs.                              [required]
  --p-n-components INTEGER
                         The underlying low-rank structure (suggested: 2 <
                         rank < 10) [minimum 2]                   [default: 3]
  --p-min-sample-count INTEGER
                         Minimum sum cutoff of sample across all features
                                                                  [default: 0]
  --p-min-feature-count INTEGER
                         Minimum sum cutoff of features across all samples
                                                                  [default: 0]
  --p-max-iterations-als INTEGER
                         Max number of Alternating Least Square (ALS)
                         optimization iterations (suggested to be below 100;
                         beware of overfitting) [minimum 1]      [default: 25]
  --p-max-iterations-rptm INTEGER
                         Max number of Robust Tensor Power Method (RTPM)
                         optimization iterations (suggested to be below 100;
                         beware of overfitting) [minimum 1]      [default: 25]
  --p-n-initializations INTEGER
                         The number of initialization vectors. Larger values
                         willgive more accurate factorization but will be more
                         computationally expensive [minimum 1]   [default: 25]
  --m-feature-metadata-file METADATA...
    (multiple            
     arguments will be   
     merged)                                                        [optional]
Outputs:
  --o-subject-biplot ARTIFACT PCoAResults % Properties('biplot')
                         Compositional biplot of subjects as points and
                         features as arrows. Where the variation between
                         subject groupings is explained by the log-ratio
                         between opposing arrows. WARNING: The % variance
                         explained is spread over n-components and can be
                         inflated.                                  [required]
  --o-state-distance-matrix ARTIFACT
    DistanceMatrix       A sample-sample distance matrix generated from the
                         euclidean distance of the subject-state ordinations
                         and itself.                                [required]
  --o-state-subject-ordination ARTIFACT SampleData[SampleTrajectory]
                         A trajectory is an ordination that can be
                         visualizedover time or another context.    [required]
  --o-state-feature-ordination ARTIFACT FeatureData[FeatureTrajectory]
                         A trajectory is an ordination that can be
                         visualizedover time or another context.    [required]
Miscellaneous:
  --output-dir PATH      Output unspecified results to a directory
  --verbose / --quiet    Display verbose output to stdout and/or stderr
                         during execution of this action. Or silence output if
                         execution is successful (silence is golden).
  --citations            Show citations and exit.
  --help                 Show this message and exit.

```

## Using gemelli as a standalone tool

```bash
$ gemelli --help

Usage: gemelli [OPTIONS]

  Runs CTF with an rclr preprocessing step.

Options:
  --in-biom TEXT                 Input table in biom format.  [required]
  --sample-metadata-file TEXT    Sample metadata file in QIIME2 formatting.
                                 [required]
  --individual-id-column TEXT    Metadata column containing subject IDs to use
                                 for pairing samples. WARNING: if replicates
                                 exist for an individual ID at either state_1
                                 to state_N, that subject will be mean grouped.
                                 [required]
  --state-column-1 TEXT          Metadata column containing state (e.g.,Time,
                                 BodySite) across which samples are paired. At
                                 least one is required but up to four are
                                 allowed by other state inputs.  [required]
  --output-dir TEXT              Location of output files.  [required]
  --n_components INTEGER         The underlying low-rank structure (suggested:
                                 1 < rank < 10) [minimum 2]  [default: 3]
  --min-sample-count INTEGER     Minimum sum cutoff of sample across all
                                 features  [default: 0]
  --min-feature-count INTEGER    Minimum sum cutoff of features across all
                                 samples  [default: 5]
  --max_iterations_als INTEGER   Max number of Alternating Least Square (ALS)
                                 optimization iterations (suggested to be
                                 below 100; beware of overfitting) [minimum 1]
                                 [default: 50]
  --max_iterations_rptm INTEGER  Max number of Robust Tensor Power Method
                                 (RTPM) optimization iterations (suggested to
                                 be below 100; beware of overfitting) [minimum
                                 1]  [default: 50]
  --n_initializations INTEGER    The number of initialization vectors. Larger
                                 values willgive more accurate factorization
                                 but will be more computationally expensive
                                 (suggested to be below 100; beware of
                                 overfitting) [minimum 1]  [default: 50]
  --feature-metadata-file TEXT   Feature metadata file in QIIME2 formatting.
  --state-column-2 TEXT          Metadata column containing state (e.g.,Time,
                                 BodySite) across which samples are paired. At
                                 least one is required but up to four are
                                 allowed by other state inputs.
  --state-column-3 TEXT          Metadata column containing state (e.g.,Time,
                                 BodySite) across which samples are paired. At
                                 least one is required but up to four are
                                 allowed by other state inputs.
  --state-column-4 TEXT          Metadata column containing state (e.g.,Time,
                                 BodySite) across which samples are paired. At
                                 least one is required but up to four are
                                 allowed by other state inputs.
  --help                         Show this message and exit.

```

## Other Resources

Named after gemelli by alighiero boetti and also the pasta. 

[TenAls translated from Sewoong Oh](http://swoh.web.engr.illinois.edu/software/optspace/code.html)
