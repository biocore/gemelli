# ----------------------------------------------------------------------------
# Copyright (c) 2019--, gemelli development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

import qiime2.plugin
import qiime2.sdk
import importlib
from gemelli import __version__
from gemelli.ctf import (ctf, phylogenetic_ctf,
                         phylogenetic_ctf_without_taxonomy,
                         phylogenetic_ctf_with_taxonomy)
from gemelli.rpca import (rpca, auto_rpca,
                          phylogenetic_rpca_with_taxonomy,
                          phylogenetic_rpca_without_taxonomy)
from gemelli.preprocessing import (rclr_transformation,
                                   phylogenetic_rclr_transformation)
from ._type import SampleTrajectory, FeatureTrajectory
from ._format import TrajectoryDirectoryFormat
from qiime2.plugin import (Properties, Int, Float, Metadata, Str)
from q2_types.ordination import PCoAResults
from q2_types.distance_matrix import DistanceMatrix
from q2_types.sample_data import SampleData
from q2_types.feature_data import FeatureData
from q2_types.tree import Phylogeny, Rooted
from q2_types.feature_table import FeatureTable, Frequency, Composition
from q2_types.feature_data import Taxonomy
from qiime2.plugin import Metadata
from gemelli._defaults import (DESC_COMP, DESC_ITERATIONSALS,
                               DESC_BIN, DESC_SMETA, DESC_TREE,
                               DESC_SUBJ, DESC_COND, DESC_INIT,
                               DESC_ITERATIONSRTPM, DESC_MINDEPTH,
                               QLOAD, QDIST, QORD, QSOAD, QRCLR,
                               DESC_MSC, DESC_MFC, QBIPLOT,
                               QTREE, QTREECOUNT, QADIST,
                               DESC_ITERATIONS, DESC_MFF, DESC_TAX_Q2,
                               DESC_T2T_TAX, DESC_STBL)

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
    output_descriptions={'rclr_table': QRCLR},
    name=('Robust centered log-ratio (rclr) transformation.'
          'Note: This is run automatically '
          'within CTF/RPCA/Auto-RPCA so there no '
          'need to run rclr before those functions.'),
    description=("A robust centered log-ratio transformation of only "
                 "the observed values (non-zero) of the input table."),
    citations=[citations['Martino2019']]
)

plugin.methods.register_function(
    function=phylogenetic_rclr_transformation,
    inputs={'table': FeatureTable[Frequency],
            'phylogeny': Phylogeny[Rooted]},
    parameters={'min_depth': Int},
    outputs=[('counts_by_node', FeatureTable[Frequency]),
             ('rclr_table', FeatureTable[Composition]),
             ('counts_by_node_tree', Phylogeny[Rooted])],
    input_descriptions={'table': DESC_BIN,
                        'phylogeny': DESC_TREE},
    parameter_descriptions={'min_depth': DESC_MINDEPTH},
    output_descriptions={'counts_by_node': QTREECOUNT,
                         'rclr_table': QRCLR,
                         'counts_by_node_tree': QTREE},
    name=('Phylogenetic Robust centered log-ratio (rclr) transformation.'
          'Note: This is run automatically '
          'within phylogenetic-CTF/RPCA so there no '
          'need to run rclr before those functions.'),
    description=("A phylogenetic robust centered log-ratio transformation "
                 "of only the observed values (non-zero) of the input table."),
    citations=[citations['Martino2019']]
)

plugin.methods.register_function(
    function=ctf,
    inputs={'table': FeatureTable[Frequency]},
    parameters={'sample_metadata': Metadata,
                'individual_id_column': Str,
                'state_column': Str,
                'n_components': Int,
                'min_sample_count': Int,
                'min_feature_count': Int,
                'min_feature_frequency': Float,
                'max_iterations_als': Int,
                'max_iterations_rptm': Int,
                'n_initializations': Int,
                'feature_metadata': Metadata},
    outputs=[('subject_biplot', PCoAResults % Properties("biplot")),
             ('state_biplot', PCoAResults % Properties("biplot")),
             ('distance_matrix', DistanceMatrix),
             ('state_subject_ordination', SampleData[SampleTrajectory]),
             ('state_feature_ordination', FeatureData[FeatureTrajectory])],
    input_descriptions={'table': DESC_BIN},
    parameter_descriptions={'sample_metadata': DESC_SMETA,
                            'individual_id_column': DESC_SUBJ,
                            'state_column': DESC_COND,
                            'n_components': DESC_COMP,
                            'min_sample_count': DESC_MSC,
                            'min_feature_count': DESC_MFC,
                            'min_feature_frequency': DESC_MFF,
                            'max_iterations_als': DESC_ITERATIONSALS,
                            'max_iterations_rptm': DESC_ITERATIONSRTPM,
                            'n_initializations': DESC_INIT},
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
    function=phylogenetic_ctf,
    inputs={'table': FeatureTable[Frequency],
            'phylogeny': Phylogeny[Rooted]},
    parameters={'sample_metadata': Metadata,
                'individual_id_column': Str,
                'state_column': Str,
                'taxonomy': Metadata,
                'n_components': Int,
                'min_sample_count': Int,
                'min_feature_count': Int,
                'min_feature_frequency': Float,
                'min_depth': Int,
                'max_iterations_als': Int,
                'max_iterations_rptm': Int,
                'n_initializations': Int},
    outputs=[('subject_biplot', PCoAResults % Properties("biplot")),
             ('state_biplot', PCoAResults % Properties("biplot")),
             ('distance_matrix', DistanceMatrix),
             ('state_subject_ordination', SampleData[SampleTrajectory]),
             ('state_feature_ordination', FeatureData[FeatureTrajectory]),
             ('counts_by_node_tree', Phylogeny[Rooted]),
             ('counts_by_node', FeatureTable[Frequency]),
             ('t2t_taxonomy', FeatureData[Taxonomy]),
             ('subject_table', FeatureTable[Frequency])],
    input_descriptions={'table': DESC_BIN,
                        'phylogeny': DESC_TREE},
    parameter_descriptions={'sample_metadata': DESC_SMETA,
                            'individual_id_column': DESC_SUBJ,
                            'state_column': DESC_COND,
                            'taxonomy': DESC_TAX_Q2,
                            'n_components': DESC_COMP,
                            'min_sample_count': DESC_MSC,
                            'min_feature_count': DESC_MFC,
                            'min_feature_frequency': DESC_MFF,
                            'max_iterations_als': DESC_ITERATIONSALS,
                            'max_iterations_rptm': DESC_ITERATIONSRTPM,
                            'n_initializations': DESC_INIT},
    output_descriptions={'subject_biplot': QLOAD,
                         'state_biplot': QSOAD,
                         'distance_matrix': QDIST,
                         'state_subject_ordination': QORD,
                         'state_feature_ordination': QORD,
                         'counts_by_node_tree': QTREE,
                         'counts_by_node': QTREECOUNT,
                         't2t_taxonomy': DESC_T2T_TAX,
                         'subject_table': DESC_STBL},
    name='Phylogenetic Compositional Tensor Factorization (CTF) '
         'with mode 3 tensor. This means subjects have repeated '
         'measures across only one axis (e.g. time or space). '
         'The input taxonomy is used to produce a new taxonomy '
         'label for each node in the tree based on the lowest '
         'common anscestor. Note: equivelent to '
         'phylogenetic-ctf-with-taxonomy',
    description=("Gemelli resolves spatiotemporal subject variation and the"
                 " biological features that separate them. In this case, a "
                 "subject may have several paired samples, where each sample"
                 " may be a time point. The output is akin to conventional "
                 "beta-diversity analyses but with the paired component "
                 "integrated in the dimensionality reduction."),
    citations=[citations['Martino2020']]
)

plugin.methods.register_function(
    function=phylogenetic_ctf_with_taxonomy,
    inputs={'table': FeatureTable[Frequency],
            'phylogeny': Phylogeny[Rooted]},
    parameters={'sample_metadata': Metadata,
                'taxonomy': Metadata,
                'individual_id_column': Str,
                'state_column': Str,
                'n_components': Int,
                'min_sample_count': Int,
                'min_feature_count': Int,
                'min_feature_frequency': Float,
                'min_depth': Int,
                'max_iterations_als': Int,
                'max_iterations_rptm': Int,
                'n_initializations': Int},
    outputs=[('subject_biplot', PCoAResults % Properties("biplot")),
             ('state_biplot', PCoAResults % Properties("biplot")),
             ('distance_matrix', DistanceMatrix),
             ('state_subject_ordination', SampleData[SampleTrajectory]),
             ('state_feature_ordination', FeatureData[FeatureTrajectory]),
             ('counts_by_node_tree', Phylogeny[Rooted]),
             ('counts_by_node', FeatureTable[Frequency]),
             ('t2t_taxonomy', FeatureData[Taxonomy]),
             ('subject_table', FeatureTable[Frequency])],
    input_descriptions={'table': DESC_BIN,
                        'phylogeny': DESC_TREE},
    parameter_descriptions={'sample_metadata': DESC_SMETA,
                            'taxonomy': DESC_TAX_Q2,
                            'individual_id_column': DESC_SUBJ,
                            'state_column': DESC_COND,
                            'n_components': DESC_COMP,
                            'min_sample_count': DESC_MSC,
                            'min_feature_count': DESC_MFC,
                            'min_feature_frequency': DESC_MFF,
                            'max_iterations_als': DESC_ITERATIONSALS,
                            'max_iterations_rptm': DESC_ITERATIONSRTPM,
                            'n_initializations': DESC_INIT},
    output_descriptions={'subject_biplot': QLOAD,
                         'state_biplot': QSOAD,
                         'distance_matrix': QDIST,
                         'state_subject_ordination': QORD,
                         'state_feature_ordination': QORD,
                         'counts_by_node_tree': QTREE,
                         'counts_by_node': QTREECOUNT,
                         't2t_taxonomy': DESC_T2T_TAX,
                         'subject_table': DESC_STBL},
    name='Phylogenetic Compositional Tensor Factorization (CTF) '
         'with mode 3 tensor. This means subjects have repeated '
         'measures across only one axis (e.g. time or space). '
         'The input taxonomy is used to produce a new taxonomy '
         'label for each node in the tree based on the lowest '
         'common anscestor.',
    description=("Gemelli resolves spatiotemporal subject variation and the"
                 " biological features that separate them. In this case, a "
                 "subject may have several paired samples, where each sample"
                 " may be a time point. The output is akin to conventional "
                 "beta-diversity analyses but with the paired component "
                 "integrated in the dimensionality reduction."),
    citations=[citations['Martino2020']]
)

plugin.methods.register_function(
    function=phylogenetic_ctf_without_taxonomy,
    inputs={'table': FeatureTable[Frequency],
            'phylogeny': Phylogeny[Rooted]},
    parameters={'sample_metadata': Metadata,
                'individual_id_column': Str,
                'state_column': Str,
                'n_components': Int,
                'min_sample_count': Int,
                'min_feature_count': Int,
                'min_feature_frequency': Float,
                'min_depth': Int,
                'max_iterations_als': Int,
                'max_iterations_rptm': Int,
                'n_initializations': Int},
    outputs=[('subject_biplot', PCoAResults % Properties("biplot")),
             ('state_biplot', PCoAResults % Properties("biplot")),
             ('distance_matrix', DistanceMatrix),
             ('state_subject_ordination', SampleData[SampleTrajectory]),
             ('state_feature_ordination', FeatureData[FeatureTrajectory]),
             ('counts_by_node_tree', Phylogeny[Rooted]),
             ('counts_by_node', FeatureTable[Frequency]),
             ('subject_table', FeatureTable[Frequency])],
    input_descriptions={'table': DESC_BIN,
                        'phylogeny': DESC_TREE},
    parameter_descriptions={'sample_metadata': DESC_SMETA,
                            'individual_id_column': DESC_SUBJ,
                            'state_column': DESC_COND,
                            'n_components': DESC_COMP,
                            'min_sample_count': DESC_MSC,
                            'min_feature_count': DESC_MFC,
                            'min_feature_frequency': DESC_MFF,
                            'max_iterations_als': DESC_ITERATIONSALS,
                            'max_iterations_rptm': DESC_ITERATIONSRTPM,
                            'n_initializations': DESC_INIT},
    output_descriptions={'subject_biplot': QLOAD,
                         'state_biplot': QSOAD,
                         'distance_matrix': QDIST,
                         'state_subject_ordination': QORD,
                         'state_feature_ordination': QORD,
                         'counts_by_node_tree': QTREE,
                         'counts_by_node': QTREECOUNT,
                         'subject_table' : DESC_STBL},
    name='Phylogenetic Compositional Tensor Factorization (CTF) '
         'with mode 3 tensor. This means subjects have repeated '
         'measures across only one axis (e.g. time or space). '
         'Note: This does not require/output a taxonomy. '
         'A taxonomy for the input phylogeny will still  be valid '
         'for tip level features however, '
         'if taxonomy is required for internal features please use '
         'phylogenetic-ctf-with-taxonomy.',
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
    parameters={'n_components': Int,
                'min_sample_count': Int,
                'min_feature_count': Int,
                'min_feature_frequency': Float,
                'max_iterations': Int},
    outputs=[('biplot', PCoAResults % Properties("biplot")),
             ('distance_matrix', DistanceMatrix)],
    input_descriptions={'table': DESC_BIN},
    parameter_descriptions={'n_components': DESC_COMP,
                            'min_sample_count': DESC_MSC,
                            'min_feature_count': DESC_MFC,
                            'min_feature_frequency': DESC_MFF,
                            'max_iterations': DESC_ITERATIONS},
    output_descriptions={'biplot': QBIPLOT,
                         'distance_matrix': QADIST},
    name='(Robust Aitchison) RPCA with manually chosen n_components.',
    description=("Performs robust center log-ratio transform "
                 "robust PCA and ranks the features by the "
                 "loadings of the resulting SVD."),
    citations=[citations['Martino2019']]
)

plugin.methods.register_function(
    function=phylogenetic_rpca_with_taxonomy,
    inputs={'table': FeatureTable[Frequency],
            'phylogeny': Phylogeny[Rooted],
            },
    parameters={
        'taxonomy': Metadata,
        'n_components': Int,
        'min_sample_count': Int,
        'min_feature_count': Int,
        'min_feature_frequency': Float,
        'min_depth': Int,
        'max_iterations': Int},
    outputs=[
        ('biplot', PCoAResults % Properties("biplot")),
        ('distance_matrix', DistanceMatrix),
        ('counts_by_node_tree', Phylogeny[Rooted]),
        ('counts_by_node', FeatureTable[Frequency]),
        ('t2t_taxonomy', FeatureData[Taxonomy])],
    input_descriptions={'table': DESC_BIN, 'phylogeny': DESC_TREE},
    parameter_descriptions={'taxonomy': DESC_TAX_Q2,
                            'n_components': DESC_COMP,
                            'min_sample_count': DESC_MSC,
                            'min_feature_count': DESC_MFC,
                            'min_feature_frequency': DESC_MFF,
                            'min_depth': DESC_MINDEPTH,
                            'max_iterations': DESC_ITERATIONS},
    output_descriptions={
        'biplot': QBIPLOT,
        'distance_matrix': QADIST,
        'counts_by_node_tree': QTREE,
        'counts_by_node': QTREECOUNT,
        't2t_taxonomy': DESC_T2T_TAX},
    name='Phylogenetic (Robust Aitchison) RPCA.',
    description=("Performs phylogenetic robust center log-ratio transform "
                 "robust PCA and ranks the features by the "
                 "loadings of the resulting SVD"),
    citations=[citations['Martino2019']]
)

plugin.methods.register_function(
    function=phylogenetic_rpca_without_taxonomy,
    inputs={'table': FeatureTable[Frequency],
            'phylogeny': Phylogeny[Rooted]},
    parameters={
        'n_components': Int,
        'min_sample_count': Int,
        'min_feature_count': Int,
        'min_feature_frequency': Float,
        'min_depth': Int,
        'max_iterations': Int},
    outputs=[
        ('biplot', PCoAResults % Properties("biplot")),
        ('distance_matrix', DistanceMatrix),
        ('counts_by_node_tree', Phylogeny[Rooted]),
        ('counts_by_node', FeatureTable[Frequency])],
    input_descriptions={'table': DESC_BIN, 'phylogeny': DESC_TREE},
    parameter_descriptions={'n_components': DESC_COMP,
                            'min_sample_count': DESC_MSC,
                            'min_feature_count': DESC_MFC,
                            'min_feature_frequency': DESC_MFF,
                            'min_depth': DESC_MINDEPTH,
                            'max_iterations': DESC_ITERATIONS},
    output_descriptions={
        'biplot': QBIPLOT,
        'distance_matrix': QADIST,
        'counts_by_node_tree': QTREE,
        'counts_by_node': QTREECOUNT},
    name=('Phylogenetic (Robust Aitchison) RPCA. '
          'Note: This does not require/output a taxonomy. '
          'A taxonomy for the input phylogeny will still  be valid '
          'for tip level features however, '
          'if taxonomy is required for internal features please use '
          'phylogenetic-rpca-with-taxonomy.'),
    description=("Performs phylogenetic robust center log-ratio transform "
                 "robust PCA and ranks the features by the "
                 "loadings of the resulting SVD"),
    citations=[citations['Martino2019']]
)

plugin.methods.register_function(
    function=auto_rpca,
    inputs={'table': FeatureTable[Frequency]},
    parameters={'min_sample_count': Int,
                'min_feature_count': Int,
                'min_feature_frequency': Float,
                'max_iterations': Int},
    outputs=[('biplot', PCoAResults % Properties("biplot")),
             ('distance_matrix', DistanceMatrix)],
    input_descriptions={'table': DESC_BIN},
    parameter_descriptions={'min_sample_count': DESC_MSC,
                            'min_feature_count': DESC_MFC,
                            'min_feature_frequency': DESC_MFF,
                            'max_iterations': DESC_ITERATIONS},
    output_descriptions={'biplot': QBIPLOT,
                         'distance_matrix': QADIST},
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
