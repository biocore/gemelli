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
from gemelli.utils import (filter_ordination)
from gemelli.q2._visualizer import qc_rarefy
from gemelli.ctf import (ctf, phylogenetic_ctf,
                         phylogenetic_ctf_without_taxonomy,
                         phylogenetic_ctf_with_taxonomy)
from gemelli.rpca import (rpca, joint_rpca,
                          rpca_with_cv,
                          feature_correlation_table,
                          phylogenetic_rpca_with_taxonomy,
                          phylogenetic_rpca_without_taxonomy,
                          transform, rpca_transform)
from gemelli.tempted import (tempted_factorize,
                             tempted_project)
from gemelli.preprocessing import (rclr_transformation,
                                   phylogenetic_rclr_transformation,
                                   clr_transformation,
                                   phylogenetic_clr_transformation)
from ._type import (SampleTrajectory, FeatureTrajectory,
                    CrossValidationResults,
                    OrdinationCorrelation)
from ._format import (TrajectoryDirectoryFormat,
                      CVDirectoryFormat,
                      CorrelationDirectoryFormat)
from qiime2.plugin import (Properties, Int, Float, Metadata,
                           Str, List, Bool, Choices, Range,
                           Visualization)
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
                               QTREE, QTREECOUNT, QADIST, QACV,
                               DESC_ITERATIONS, DESC_MFF, DESC_TAX_Q2,
                               DESC_T2T_TAX, DESC_STBL, DESC_METACV,
                               DESC_TABLES, DESC_COLCV, DESC_TESTS,
                               DESC_TABLES, DESC_MATCH, 
                               DEFAULT_TRNSFRM, DESC_TRNSFRM,
                               DESC_TRAINTABLES, DESC_TRAINORDS,
                               DESC_MTABLE, DESC_MORD, DESC_FM,
                               DESC_SM, DESC_MORDOUT,
                               DESC_CORRTBLORD, DESC_CORRTBL,
                               DESC_TCOND, DESC_REP, DESC_SVD,
                               DESC_SVDC, DESC_SMTH, DESC_RES,
                               DESC_MXTR, DESC_EPS, DESC_IO,
                               DESC_SLO, DESC_TDIST, DESC_SVDO,
                               DESC_PIO, DESC_PC, DESC_TJNT)

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
          ' This is run automatically '
          'within CTF/RPCA so there no '
          'need to run rclr before those functions.'),
    description=("A robust centered log-ratio transformation of only "
                 "the observed values (non-zero) of the input table."),
    citations=[citations['Martino2019']]
)

plugin.methods.register_function(
    function=clr_transformation,
    inputs={'table': FeatureTable[Frequency]},
    parameters={'pseudocount': Float},
    outputs=[('clr_table', FeatureTable[Composition])],
    input_descriptions={'table': DESC_BIN},
    parameter_descriptions={'pseudocount': DESC_PC},
    output_descriptions={'clr_table': QRCLR},
    name=('Centered log-ratio (clr) transformation.'
          'By default a pseudocount is added with the minimum '
          'non-zero value.'),
    description=("A centered log-ratio transformation "
                 "of the input table."),
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
    function=phylogenetic_clr_transformation,
    inputs={'table': FeatureTable[Frequency],
            'phylogeny': Phylogeny[Rooted]},
    parameters={'min_depth': Int,
                'pseudocount': Float},
    outputs=[('counts_by_node', FeatureTable[Frequency]),
             ('clr_table', FeatureTable[Composition]),
             ('counts_by_node_tree', Phylogeny[Rooted])],
    input_descriptions={'table': DESC_BIN,
                        'phylogeny': DESC_TREE},
    parameter_descriptions={'min_depth': DESC_MINDEPTH,
                            'pseudocount': DESC_PC},
    output_descriptions={'counts_by_node': QTREECOUNT,
                         'clr_table': QRCLR,
                         'counts_by_node_tree': QTREE},
    name=('Phylogenetic centered log-ratio (clr) transformation.'),
    description=("A phylogenetic centered log-ratio transformation "
                 "By default a pseudocount is added with the minimum "
                 "non-zero value."),
    citations=[citations['Martino2019']]
)

plugin.methods.register_function(
    function=tempted_factorize,
    inputs={'table': FeatureTable[Composition]},
    parameters={'sample_metadata': Metadata,
                'individual_id_column': Str,
                'state_column': Str,
                'n_components': Int,
                'replicate_handling': Str,
                'svd_centralized': Bool,
                'n_components_centralize': Int,
                'smooth': Float,
                'resolution': Int,
                'max_iterations': Int,
                'epsilon': Float},
    outputs=[('individual_biplot', PCoAResults % Properties("biplot")),
             ('state_loadings', SampleData[SampleTrajectory]),
             ('distance_matrix', DistanceMatrix),
             ('svd_center', SampleData[SampleTrajectory])],
    input_descriptions={'table': DESC_BIN},
    parameter_descriptions={'sample_metadata': DESC_SMETA,
                            'individual_id_column': DESC_SUBJ,
                            'state_column': DESC_TCOND,
                            'n_components': DESC_COMP,
                            'replicate_handling': DESC_REP,
                            'svd_centralized': DESC_SVD,
                            'n_components_centralize': DESC_SVDC,
                            'smooth': DESC_SMTH,
                            'resolution': DESC_RES,
                            'max_iterations': DESC_MXTR,
                            'epsilon': DESC_EPS},
    output_descriptions={'individual_biplot': DESC_IO,
                         'state_loadings': DESC_SLO,
                         'distance_matrix': DESC_TDIST,
                         'svd_center': DESC_SVDO},
    name='TEMPTED temporal tensor factorization.',
    description=("Decomposition of temporal tensors."),
    citations=[citations['Martino2020']]
)

plugin.methods.register_function(
    function=tempted_project,
    inputs={'individual_biplot': PCoAResults % Properties("biplot"),
            'state_loadings': SampleData[SampleTrajectory],
            'svd_center': SampleData[SampleTrajectory],
            'table': FeatureTable[Composition]},
    parameters={'sample_metadata': Metadata,
                'individual_id_column': Str,
                'state_column': Str,
                'replicate_handling': Str},
    outputs=[('individual_biplot', PCoAResults % Properties("biplot"))],
    input_descriptions={'table': DESC_BIN},
    parameter_descriptions={'sample_metadata': DESC_SMETA,
                            'individual_id_column': DESC_SUBJ,
                            'state_column': DESC_TCOND,
                            'replicate_handling': DESC_REP},
    output_descriptions={'individual_biplot': DESC_PIO},
    name='TEMPTED projection or new data into the subject space.',
    description=("Projection of new unseen temporal data to the low-dim"
                 " subject loading build on previous data. Warning: Ensure"
                 " the pre-processing parameters are the same as those use to"
                 "build the original results or the projection may "
                 "be spurious."),
    citations=[citations['Martino2020']]
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
    function=rpca_with_cv,
    inputs={'table': FeatureTable[Frequency]},
    parameters={'n_test_samples': Int,
                'sample_metadata': Metadata,
                'train_test_column': Str,
                'n_components': Int,
                'min_sample_count': Int,
                'min_feature_count': Int,
                'min_feature_frequency': Float,
                'max_iterations': Int},
    outputs=[('biplot', PCoAResults % Properties("biplot")),
             ('distance_matrix', DistanceMatrix),
             ('cross_validation_error', SampleData[CrossValidationResults])],
    input_descriptions={'table': DESC_BIN},
    parameter_descriptions={'n_test_samples':DESC_TESTS,
                            'sample_metadata':DESC_METACV,
                            'train_test_column':DESC_COLCV,
                            'n_components': DESC_COMP,
                            'min_sample_count': DESC_MSC,
                            'min_feature_count': DESC_MFC,
                            'min_feature_frequency': DESC_MFF,
                            'max_iterations': DESC_ITERATIONS},
    output_descriptions={'biplot': QBIPLOT,
                         'distance_matrix': QADIST,
                         'cross_validation_error': QACV},
    name='(Robust Aitchison) RPCA with manually chosen n_components.'
         ' with cross-validation output.',
    description=("Performs robust center log-ratio transform "
                 "robust PCA and ranks the features by the "
                 "loadings of the resulting SVD."),
    citations=[citations['Martino2019']]
)

plugin.methods.register_function(
    function=joint_rpca,
    inputs={'tables': List[FeatureTable[Frequency]]},
    parameters={'n_test_samples': Int,
                'sample_metadata': Metadata,
                'train_test_column': Str,
                'n_components': Int,
                'rclr_transform_tables': Bool,
                'min_sample_count': Int,
                'min_feature_count': Int,
                'min_feature_frequency': Float,
                'max_iterations': Int},
    outputs=[('biplot', PCoAResults % Properties("biplot")),
             ('distance_matrix', DistanceMatrix),
             ('cross_validation_error', SampleData[CrossValidationResults])],
    input_descriptions={'tables': DESC_TABLES},
    parameter_descriptions={'n_test_samples':DESC_TESTS,
                            'sample_metadata':DESC_METACV,
                            'train_test_column':DESC_COLCV,
                            'n_components': DESC_COMP,
                            'rclr_transform_tables':DESC_TJNT,
                            'min_sample_count': DESC_MSC,
                            'min_feature_count': DESC_MFC,
                            'min_feature_frequency': DESC_MFF,
                            'max_iterations': DESC_ITERATIONS},
    output_descriptions={'biplot': QBIPLOT,
                         'distance_matrix': QADIST,
                         'cross_validation_error': QACV},
    name='Joint (Robust Aitchison) RPCA with manually chosen n_components.',
    description=("Performs robust center log-ratio transform "
                 "joint robust PCA and ranks the features by the "
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
    function=transform,
    inputs={'ordination': PCoAResults % Properties("biplot"),
            'tables': List[FeatureTable[Frequency]]},
    parameters={'subset_tables': Bool,
                'rclr_transform': Bool % Choices(DEFAULT_TRNSFRM)},
    outputs=[('projected_biplot', PCoAResults % Properties("biplot"))],
    input_descriptions={'ordination': DESC_TRAINORDS,
                        'tables': DESC_TRAINTABLES},
    parameter_descriptions={'subset_tables': DESC_MATCH,
                            'rclr_transform': DESC_TRNSFRM},
    output_descriptions={'projected_biplot': QBIPLOT},
    name='Project dimensionality reduction to new table(s).',
    description=("Apply dimensionality reduction to table(s). The table(s)"
                 " is projected on the first principal components"
                 "previously extracted from a training set."
                 " This function works from output of RPCA with"
                 " one table as input or"
                 " Joint-RPCA but not yet phylo-RPCA."),
    citations=[citations['Martino2019']]
)

plugin.visualizers.register_function(
    function=qc_rarefy,
    inputs={'table': FeatureTable[Frequency],
            'rarefied_distance': DistanceMatrix,
            'unrarefied_distance': DistanceMatrix},
    parameters={'permutations': Int % Range(0, None)},
    input_descriptions={
        'table': 'Unrarefied table used to generate the distance matrix.',
        'rarefied_distance': 'Distance matrix produced from rarefied table.',
        'unrarefied_distance': 'Distance matrix produced from rarefied table '
                               'with the same IDs as the unrarefied table.'
    },
    parameter_descriptions={
        'permutations': 'The number of permutations to be run when computing '
                        'p-values. Supplying a value of zero will disable '
                        'permutation testing and p-values will not be '
                        'calculated (this results in *much* quicker execution '
                        'time if p-values are not desired).',
    },
    name='QC distance and abs. sample sum difference correlation '
         'with and without rarefaction.',
    description=('Determine whether the sample distances are '
                 'correlated with the sample sum differences. '
                 'With and without rarefaction before using RPCA.'),
    citations=[],
)

plugin.methods.register_function(
    function=rpca_transform,
    inputs={'ordination': PCoAResults % Properties("biplot"),
            'table': FeatureTable[Frequency]},
    parameters={'subset_tables': Bool,
                'rclr_transform': Bool % Choices(DEFAULT_TRNSFRM)},
    outputs=[('projected_biplot', PCoAResults % Properties("biplot"))],
    input_descriptions={'ordination': DESC_TRAINORDS,
                        'table': DESC_TRAINTABLES},
    parameter_descriptions={'subset_tables': DESC_MATCH,
                            'rclr_transform': DESC_TRNSFRM},
    output_descriptions={'projected_biplot': QBIPLOT},
    name='Project dimensionality reduction to a new table.',
    description=("Apply dimensionality reduction to a table. The table"
                 " is projected on the first principal components"
                 "previously extracted from a training set."
                 " This function works from output of RPCA only."),
    citations=[citations['Martino2019']]
)

plugin.methods.register_function(
    function=filter_ordination,
    inputs={'ordination': PCoAResults % Properties("biplot"),
            'table': FeatureTable[Frequency]},
    parameters={'match_features': Bool,
                'match_samples': Bool},
    outputs=[('subset_biplot', PCoAResults % Properties("biplot"))],
    input_descriptions={'ordination': DESC_MORD,
                        'table': DESC_MTABLE},
    parameter_descriptions={'match_features': DESC_FM,
                            'match_samples': DESC_SM},
    output_descriptions={'subset_biplot': DESC_MORDOUT},
    name='Filter a biplot ordination to a tables samples & features.',
    description=("Subsets an OrdinationResults to only those"
                 " samples and features shared with the input table."),
    citations=[citations['Martino2019']]
)

plugin.methods.register_function(
    function=feature_correlation_table,
    inputs={'ordination': PCoAResults % Properties("biplot")},
    parameters={},
    outputs=[('correlation_table', FeatureData[OrdinationCorrelation])],
    input_descriptions={'ordination': DESC_CORRTBLORD},
    parameter_descriptions={},
    output_descriptions={'correlation_table': DESC_CORRTBL},
    name='Generates a feature-by-feature correlation table.',
    description=("Produces a feature by feature correlation table from"
                 " Joint-RPCA/RPCA ordination results. Note that the"
                 " output can be very large in file size because it"
                 " is all omics features by all omics features and"
                 " is fully dense. If you would like to get a subset,"
                 " just subset the ordination with the function "
                 "`filter_ordination` in utils first."),
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

plugin.register_semantic_types(CrossValidationResults)
plugin.register_semantic_type_to_format(
    SampleData[CrossValidationResults],
    artifact_format=CVDirectoryFormat)
plugin.register_formats(CVDirectoryFormat)

plugin.register_semantic_types(OrdinationCorrelation)
plugin.register_semantic_type_to_format(
    FeatureData[OrdinationCorrelation],
    artifact_format=CorrelationDirectoryFormat)
plugin.register_formats(CorrelationDirectoryFormat)

importlib.import_module('gemelli.q2._transformer')
