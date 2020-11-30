# ----------------------------------------------------------------------------
# Copyright (c) 2016--, deicode development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

import qiime2.plugin
import qiime2.sdk
import importlib
from gemelli import __version__
from gemelli.ctf import ctf
from gemelli.rpca import rpca, auto_rpca
from gemelli.preprocessing import rclr_transformation
from ._type import SampleTrajectory, FeatureTrajectory
from ._format import TrajectoryDirectoryFormat
from qiime2.plugin import (Properties, Int, Float, Metadata, Str)
from q2_types.ordination import PCoAResults
from q2_types.distance_matrix import DistanceMatrix
from q2_types.sample_data import SampleData
from q2_types.feature_data import FeatureData
from q2_types.feature_table import FeatureTable, Frequency, Composition
from gemelli._defaults import (DESC_COMP, DESC_ITERATIONSALS,
                               DESC_BIN, DESC_SMETA,
                               DESC_SUBJ, DESC_COND, DESC_INIT,
                               DESC_ITERATIONSRTPM,
                               QLOAD, QDIST, QORD, QSOAD,
                               DESC_MSC, DESC_MFC,
                               DESC_ITERATIONS, DESC_MFF)

PARAMETERS = {'sample_metadata': Metadata,
              'individual_id_column': Str,
              'state_column': Str,
              'n_components': Int,
              'min_sample_count': Int,
              'min_feature_count': Int,
              'max_iterations_als': Int,
              'max_iterations_rptm': Int,
              'n_initializations': Int,
              'feature_metadata': Metadata}
PARAMETERDESC = {'sample_metadata': DESC_SMETA,
                 'individual_id_column': DESC_SUBJ,
                 'state_column': DESC_COND,
                 'n_components': DESC_COMP,
                 'min_sample_count': DESC_MSC,
                 'min_feature_count': DESC_MFC,
                 'max_iterations_als': DESC_ITERATIONSALS,
                 'max_iterations_rptm': DESC_ITERATIONSRTPM,
                 'n_initializations': DESC_INIT}

citations = qiime2.plugin.Citations.load(
    'citations.bib', package='gemelli')

plugin = qiime2.plugin.Plugin(
    name='gemelli',
    version=__version__,
    website="https://github.com/biocore/gemelli",
    citations=[citations['Martino2019'],
               citations['Martino2020']],
    short_description=('Plugin for Compositional Tensor Factorization'),
    description=('This is a QIIME 2 plugin supporting Robust Aitchison on '
                 'feature tables'),
    package='gemelli')

plugin.methods.register_function(
    function=rclr_transformation,
    inputs={'table': FeatureTable[Frequency]},
    parameters=None,
    outputs=[('rclr_table', FeatureTable[Composition])],
    input_descriptions={'table': DESC_BIN},
    parameter_descriptions=None,
    output_descriptions={'rclr_table': 'A rclr transformed table. '
                                       'Note: zero/missing values have NaNs'},
    name='Robust centered log-ratio (rclr) transformation.'
         'Note: This is run automatically '
         ' within CTF/RPCA/Auto-RPCA so there no '
         'need to run rclr before those functions.',
    description=("A robust centered log-ratio transformation of only "
                 "the observed values (non-zero) of the input table."),
    citations=[citations['Martino2019']]
)

plugin.methods.register_function(
    function=ctf,
    inputs={'table': FeatureTable[Frequency]},
    parameters=PARAMETERS,
    outputs=[('subject_biplot', PCoAResults % Properties("biplot")),
             ('state_biplot', PCoAResults % Properties("biplot")),
             ('distance_matrix', DistanceMatrix),
             ('state_subject_ordination', SampleData[SampleTrajectory]),
             ('state_feature_ordination', FeatureData[FeatureTrajectory])],
    input_descriptions={'table': DESC_BIN},
    parameter_descriptions=PARAMETERDESC,
    output_descriptions={'subject_biplot': QLOAD,
                         'state_biplot': QSOAD,
                         'distance_matrix': QDIST,
                         'state_subject_ordination': QORD,
                         'state_feature_ordination': QORD},
    name='Compositional Tensor Factorization (CTF) with mode 3 tensor. This '
         'means subjects have repeated measures across only one '
         'axis (e.g. time or space).',
    description=("Gemelli resolves spatiotemporal subject variation and the"
                 " biological features that separate them. In this case, a "
                 "subject may have several paired samples, where each sample"
                 " may be a time point. The output is akin to conventional "
                 "beta-diversity analyses but with the paired component "
                 "integrated in the dimensionality reduction."),
    citations=[citations['Martino2020']]
)

plugin.methods.register_function(
    function=rpca,
    inputs={'table': FeatureTable[Frequency]},
    parameters={
        'n_components': Int,
        'min_sample_count': Int,
        'min_feature_count': Int,
        'min_feature_frequency': Float,
        'max_iterations': Int,
    },
    outputs=[
        ('biplot', PCoAResults % Properties("biplot")),
        ('distance_matrix', DistanceMatrix)
    ],
    input_descriptions={
        'table': 'Input table of counts.',
    },
    parameter_descriptions={
        'n_components': DESC_COMP,
        'min_sample_count': DESC_MSC,
        'min_feature_count': DESC_MFC,
        'min_feature_frequency': DESC_MFF,
        'max_iterations': DESC_ITERATIONS,
    },
    output_descriptions={
        'biplot': ('A biplot of the (Robust Aitchison) RPCA feature loadings'),
        'distance_matrix': ('The Aitchison distance of'
                            ' the sample loadings from RPCA.')
    },
    name='(Robust Aitchison) RPCA with manually chosen n_components.',
    description=("Performs robust center log-ratio transform "
                 "robust PCA and ranks the features by the "
                 "loadings of the resulting SVD."),
    citations=[citations['Martino2019']]
)

plugin.methods.register_function(
    function=auto_rpca,
    inputs={'table': FeatureTable[Frequency]},
    parameters={
        'min_sample_count': Int,
        'min_feature_count': Int,
        'min_feature_frequency': Float,
        'max_iterations': Int,
    },
    outputs=[
        ('biplot', PCoAResults % Properties("biplot")),
        ('distance_matrix', DistanceMatrix)
    ],
    input_descriptions={
        'table': 'Input table of counts.',
    },
    parameter_descriptions={
        'min_sample_count': DESC_MSC,
        'min_feature_count': DESC_MFC,
        'min_feature_frequency': DESC_MFF,
        'max_iterations': DESC_ITERATIONS,
    },
    output_descriptions={
        'biplot': ('A biplot of the (Robust Aitchison) RPCA feature loadings'),
        'distance_matrix': ('The Aitchison distance of'
                            ' the sample loadings from RPCA.')
    },
    name='(Robust Aitchison) RPCA with n_components automatically detected.',
    description=("Performs robust center log-ratio transform "
                 "robust PCA and ranks the features by the "
                 "loadings of the resulting SVD. Automatically"
                 " estimates the underlying rank (i.e. n-components)."),
    citations=[citations['Martino2019']]
)

plugin.register_semantic_types(SampleTrajectory, FeatureTrajectory)
plugin.register_semantic_type_to_format(
    SampleData[SampleTrajectory],
    artifact_format=TrajectoryDirectoryFormat)
plugin.register_semantic_type_to_format(
    FeatureData[FeatureTrajectory],
    artifact_format=TrajectoryDirectoryFormat)
plugin.register_formats(TrajectoryDirectoryFormat)
importlib.import_module('gemelli.q2._transformer')
