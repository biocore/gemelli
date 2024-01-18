import os
import click
import pandas as pd
from .__init__ import cli
from biom import load_table
from biom.util import biom_open
from skbio import OrdinationResults
from gemelli.preprocessing import TaxonomyError
from gemelli.utils import filter_ordination as _filter_ordination
from gemelli.rpca import rpca as _rpca
from gemelli.rpca import rpca_with_cv as _rpca_with_cv
from gemelli.rpca import (feature_correlation_table as 
                          _feature_correlation_table)
from gemelli.rpca import phylogenetic_rpca as _phylo_rpca
from gemelli.rpca import joint_rpca as _joint_rpca
from gemelli.rpca import transform as _transform
from gemelli._defaults import (DEFAULT_COMP, DEFAULT_MSC, DEFAULT_MTD,
                               DEFAULT_MFC, DEFAULT_OPTSPACE_ITERATIONS,
                               DESC_COMP, DESC_MSC, DESC_MFC,
                               DESC_ITERATIONS, DESC_MINDEPTH,
                               DEFAULT_MFF, DESC_MFF,
                               DESC_COUNTS, DESC_TREE, DESC_TAX_SA,
                               DESC_METACV, DESC_COLCV, DESC_TESTS,
                               DEFAULT_METACV, DEFAULT_COLCV,
                               DEFAULT_TESTS, DESC_TABLES,
                               DEFAULT_MATCH, DESC_MATCH,
                               DESC_TRAINTABLES, DESC_TRAINORDS,
                               DESC_TRAINTABLE, DESC_TRAINORD,
                               DESC_MTABLE, DESC_MORD, DESC_FM,
                               DESC_SM, DESC_CORRTBLORD,
                               DESC_TJNT, DEFAULT_TRNSFRM)

@cli.command(name='phylogenetic-rpca')
@click.option('--in-biom',
              help=DESC_COUNTS,
              required=True)
@click.option('--in-phylogeny',
              help=DESC_TREE,
              required=True)
@click.option('--taxonomy',
              default=None,
              show_default=True,
              help=DESC_TAX_SA)
@click.option('--output-dir',
              help='Location of output files.',
              required=True)
@click.option('--n-components',
              default=DEFAULT_COMP,
              show_default=True,
              help=DESC_COMP)
@click.option('--min-sample-count',
              default=DEFAULT_MSC,
              show_default=True,
              help=DESC_MSC)
@click.option('--min-feature-count',
              default=DEFAULT_MFC,
              show_default=True,
              help=DESC_MFC)
@click.option('--min-feature-frequency',
              default=DEFAULT_MFF,
              show_default=True,
              help=DESC_MFF)
@click.option('--min-depth',
              default=DEFAULT_MTD,
              show_default=True,
              help=DESC_MINDEPTH)
@click.option('--max-iterations',
              default=DEFAULT_OPTSPACE_ITERATIONS,
              show_default=True,
              help=DESC_ITERATIONS)
def standalone_phylogenetic_rpca(in_biom: str,
                                 in_phylogeny: str,
                                 taxonomy: None,
                                 output_dir: str,
                                 n_components: int,
                                 min_sample_count: int,
                                 min_feature_count: int,
                                 min_feature_frequency: float,
                                 min_depth: int,
                                 max_iterations: int) -> None:
    """Runs phylogenetically informed RPCA with an rclr preprocessing step."""

    # import table
    table = load_table(in_biom)
    # import taxonomy
    taxonomy_table = None
    if taxonomy is not None:
        taxonomy_table = pd.read_csv(taxonomy, sep='\t')
        try:
            taxonomy_table.set_index('Feature ID', inplace=True)
        except KeyError:
            raise TaxonomyError(
                        "Taxonomy file must have a column labled 'Feature ID'."
                        )
    # run the RPCA wrapper
    phylo_res_ = _phylo_rpca(table,
                             in_phylogeny,
                             taxonomy_table,
                             n_components,
                             min_sample_count,
                             min_feature_count,
                             min_feature_frequency,
                             min_depth,
                             max_iterations)
    ord_res, dist_res, phylogeny, counts_by_node, result_taxonomy = phylo_res_
    # If it doesn't already exist, create the output directory.
    # Note that there is technically a race condition here: it's ostensibly
    # possible that some process could delete the output directory after we
    # check that it exists here but before we write the output files to it.
    # However, in this case, we'd just get an error from skbio.io.util.open()
    # (which is called by skbio.OrdinationResults.write()), which makes sense.
    os.makedirs(output_dir, exist_ok=True)

    # write files to output directory
    # Note that this will overwrite files in the output directory that share
    # these filenames (analogous to QIIME 2's behavior if you specify the
    # --o-biplot and --o-distance-matrix options, but differing from QIIME 2's
    # behavior if you specify --output-dir instead).
    ord_res.write(os.path.join(output_dir, 'ordination.txt'))
    dist_res.write(os.path.join(output_dir, 'distance-matrix.tsv'))
    phylogeny.write(os.path.join(output_dir, 'labeled-phylogeny.nwk'))
    if result_taxonomy is not None:
        result_taxonomy.to_csv(os.path.join(output_dir, 't2t-taxonomy.tsv'),
                           sep='\t')
    # write the vectorized count table for Qurro / log-ratios
    with biom_open(os.path.join(output_dir, 'phylo-table.biom'), 'w') as f:
        counts_by_node.to_hdf5(f, "phylo-rpca-count-table")


@cli.command(name='joint-rpca')
@click.option('--in-biom',
              help=DESC_TABLES,
              required=True,
              multiple=True)
@click.option('--output-dir',
              help='Location of output files.',
              required=True)
@click.option('--n-test-samples',
              default=DEFAULT_TESTS,
              show_default=True,
              help=DESC_TESTS)
@click.option('--sample-metadata-file',
              default=DEFAULT_METACV,
              show_default=True,
              help=DESC_METACV)
@click.option('--train-test-column',
              default=DEFAULT_COLCV,
              show_default=True,
              help=DESC_COLCV)
@click.option('--n-components',
              default=DEFAULT_COMP,
              show_default=True,
              help=DESC_COMP)
@click.option('--no-rclr-transform-tables',
              default=False,
              show_default=True,
              is_flag=True,
              help=DESC_TJNT)
@click.option('--min-sample-count',
              default=DEFAULT_MSC,
              show_default=True,
              help=DESC_MSC)
@click.option('--min-feature-count',
              default=DEFAULT_MFC,
              show_default=True,
              help=DESC_MFC)
@click.option('--min-feature-frequency',
              default=DEFAULT_MFF,
              show_default=True,
              help=DESC_MFF)
@click.option('--max-iterations',
              default=DEFAULT_OPTSPACE_ITERATIONS,
              show_default=True,
              help=DESC_ITERATIONS)
def standalone_joint_rpca(in_biom: list,
                          output_dir: str,
                          n_test_samples: int,
                          sample_metadata_file: str,
                          train_test_column: str,
                          n_components: int,
                          no_rclr_transform_tables: bool,
                          min_sample_count: int,
                          min_feature_count: int,
                          min_feature_frequency: float,
                          max_iterations: int) -> None:
    """Runs Joint-RPCA."""

    # import tables
    if isinstance(in_biom, list) or isinstance(in_biom, tuple):
        tables = [load_table(table) for table in in_biom]
    else:
        tables = [load_table(in_biom)]
    if sample_metadata_file is not None:
        # import sample metadata (if needed)
        sample_metadata = pd.read_csv(sample_metadata_file,
                                      sep='\t', index_col=0,
                                      low_memory=False)
    else:
        sample_metadata = None
    # run the RPCA wrapper
    rclr_transform_tables = not no_rclr_transform_tables
    res_tmp = _joint_rpca(tables,
                          n_test_samples=n_test_samples,
                          sample_metadata=sample_metadata,
                          train_test_column=train_test_column,
                          n_components=n_components,
                          rclr_transform_tables=rclr_transform_tables,
                          min_sample_count=min_sample_count,
                          min_feature_count=min_feature_count,
                          min_feature_frequency=min_feature_frequency,
                          max_iterations=max_iterations)
    ord_res, dist_res, cv_res = res_tmp
    # If it doesn't already exist, create the output directory.
    # Note that there is technically a race condition here: it's ostensibly
    # possible that some process could delete the output directory after we
    # check that it exists here but before we write the output files to it.
    # However, in this case, we'd just get an error from skbio.io.util.open()
    # (which is called by skbio.OrdinationResults.write()), which makes sense.
    os.makedirs(output_dir, exist_ok=True)

    # write files to output directory
    # Note that this will overwrite files in the output directory that share
    # these filenames (analogous to QIIME 2's behavior if you specify the
    # --o-biplot and --o-distance-matrix options, but differing from QIIME 2's
    # behavior if you specify --output-dir instead).
    ord_res.write(os.path.join(output_dir, 'joint-ordination.txt'))
    dist_res.write(os.path.join(output_dir, 'joint-distance-matrix.tsv'))
    cv_res.to_csv(os.path.join(output_dir, 'cross-validation-error.tsv'), sep='\t')


@cli.command(name='rpca-with-cv')
@click.option('--in-biom',
              help=DESC_COUNTS,
              required=True)
@click.option('--output-dir',
              help='Location of output files.',
              required=True)
@click.option('--n-test-samples',
              default=DEFAULT_TESTS,
              show_default=True,
              help=DESC_TESTS)
@click.option('--sample-metadata-file',
              default=DEFAULT_METACV,
              show_default=True,
              help=DESC_METACV)
@click.option('--train-test-column',
              default=DEFAULT_COLCV,
              show_default=True,
              help=DESC_COLCV)
@click.option('--n-components',
              default=DEFAULT_COMP,
              show_default=True,
              help=DESC_COMP)
@click.option('--min-sample-count',
              default=DEFAULT_MSC,
              show_default=True,
              help=DESC_MSC)
@click.option('--min-feature-count',
              default=DEFAULT_MFC,
              show_default=True,
              help=DESC_MFC)
@click.option('--min-feature-frequency',
              default=DEFAULT_MFF,
              show_default=True,
              help=DESC_MFF)
@click.option('--max-iterations',
              default=DEFAULT_OPTSPACE_ITERATIONS,
              show_default=True,
              help=DESC_ITERATIONS)
def standalone_rpca_with_cv(in_biom: str,
                            output_dir: str,
                            n_test_samples: int,
                            sample_metadata_file: str,
                            train_test_column: str,
                            n_components: int,
                            min_sample_count: int,
                            min_feature_count: int,
                            min_feature_frequency: float,
                            max_iterations: int) -> None:
    """Runs RPCA with an rclr preprocessing step and CV."""
    # import table
    table = load_table(in_biom)
    if sample_metadata_file is not None:
        # import sample metadata (if needed)
        sample_metadata = pd.read_csv(sample_metadata_file,
                                      sep='\t', index_col=0,
                                      low_memory=False)
    else:
        sample_metadata = None
    # run the RPCA wrapper
    res_tmp = _rpca_with_cv(table,
                            n_test_samples=n_test_samples,
                            sample_metadata=sample_metadata,
                            train_test_column=train_test_column,
                            n_components=n_components,
                            min_sample_count=min_sample_count,
                            min_feature_count=min_feature_count,
                            min_feature_frequency=min_feature_frequency,
                            max_iterations=max_iterations)
    ord_res, dist_res, cv_res = res_tmp
    # If it doesn't already exist, create the output directory.
    # Note that there is technically a race condition here: it's ostensibly
    # possible that some process could delete the output directory after we
    # check that it exists here but before we write the output files to it.
    # However, in this case, we'd just get an error from skbio.io.util.open()
    # (which is called by skbio.OrdinationResults.write()), which makes sense.
    os.makedirs(output_dir, exist_ok=True)
    # write files to output directory
    # Note that this will overwrite files in the output directory that share
    # these filenames (analogous to QIIME 2's behavior if you specify the
    # --o-biplot and --o-distance-matrix options, but differing from QIIME 2's
    # behavior if you specify --output-dir instead).
    ord_res.write(os.path.join(output_dir, 'ordination.txt'))
    dist_res.write(os.path.join(output_dir, 'distance-matrix.tsv'))
    cv_res.to_csv(os.path.join(output_dir, 'cross-validation-error.tsv'), sep='\t')


@cli.command(name='rpca')
@click.option('--in-biom',
              help=DESC_COUNTS,
              required=True)
@click.option('--output-dir',
              help='Location of output files.',
              required=True)
@click.option('--n-components',
              default=DEFAULT_COMP,
              show_default=True,
              help=DESC_COMP)
@click.option('--min-sample-count',
              default=DEFAULT_MSC,
              show_default=True,
              help=DESC_MSC)
@click.option('--min-feature-count',
              default=DEFAULT_MFC,
              show_default=True,
              help=DESC_MFC)
@click.option('--min-feature-frequency',
              default=DEFAULT_MFF,
              show_default=True,
              help=DESC_MFF)
@click.option('--max-iterations',
              default=DEFAULT_OPTSPACE_ITERATIONS,
              show_default=True,
              help=DESC_ITERATIONS)
def standalone_rpca(in_biom: str,
                    output_dir: str,
                    n_components: int,
                    min_sample_count: int,
                    min_feature_count: int,
                    min_feature_frequency: float,
                    max_iterations: int) -> None:
    """Runs RPCA with an rclr preprocessing step."""

    # import table
    table = load_table(in_biom)
    # run the RPCA wrapper
    ord_res, dist_res = _rpca(table,
                              n_components,
                              min_sample_count,
                              min_feature_count,
                              min_feature_frequency,
                              max_iterations)

    # If it doesn't already exist, create the output directory.
    # Note that there is technically a race condition here: it's ostensibly
    # possible that some process could delete the output directory after we
    # check that it exists here but before we write the output files to it.
    # However, in this case, we'd just get an error from skbio.io.util.open()
    # (which is called by skbio.OrdinationResults.write()), which makes sense.
    os.makedirs(output_dir, exist_ok=True)

    # write files to output directory
    # Note that this will overwrite files in the output directory that share
    # these filenames (analogous to QIIME 2's behavior if you specify the
    # --o-biplot and --o-distance-matrix options, but differing from QIIME 2's
    # behavior if you specify --output-dir instead).
    ord_res.write(os.path.join(output_dir, 'ordination.txt'))
    dist_res.write(os.path.join(output_dir, 'distance-matrix.tsv'))


@cli.command(name='rpca-transform')
@click.option('--in-ordination',
              help=DESC_TRAINORD,
              required=True)
@click.option('--in-biom',
              help=DESC_TRAINTABLE,
              required=True)
@click.option('--output-dir',
              help='Location of output files.',
              required=True)
@click.option('--subset-tables',
              default=DEFAULT_MATCH,
              show_default=True,
              help=DESC_MATCH)
def rpca_transform(in_ordination: str,
                   in_biom: str,
                   output_dir: str,
                   subset_tables: bool) -> None:
    """ 
    Apply dimensionality reduction to table.
    The table is projected on the first principal components
    previously extracted from a training set.
    """
    # import data
    table = load_table(in_biom)
    ordination = OrdinationResults.read(in_ordination)
    # apply the transformation
    ord_res = _transform(ordination, [table],
                         subset_tables=subset_tables)
    # If it doesn't already exist, create the output directory.
    # Note that there is technically a race condition here: it's ostensibly
    # possible that some process could delete the output directory after we
    # check that it exists here but before we write the output files to it.
    # However, in this case, we'd just get an error from skbio.io.util.open()
    # (which is called by skbio.OrdinationResults.write()), which makes sense.
    os.makedirs(output_dir, exist_ok=True)
    # write results
    ord_res.write(os.path.join(output_dir, 'projected-ordination.txt'))


@cli.command(name='joint-rpca-transform')
@click.option('--in-ordination',
              help=DESC_TRAINORDS,
              required=True)
@click.option('--in-biom',
              help=DESC_TRAINTABLES,
              required=True,
              multiple=True)
@click.option('--output-dir',
              help='Location of output files.',
              required=True)
@click.option('--subset-tables',
              default=DEFAULT_MATCH,
              show_default=True,
              help=DESC_MATCH)
def joint_pca_transform(in_ordination: str,
                        in_biom: str,
                        output_dir: str,
                        subset_tables: bool) -> None:
    """ 
    Apply dimensionality reduction to tables.
    The tables is projected on the first principal components
    previously extracted from a training set.
    """

    # import tables
    if isinstance(in_biom, list) or isinstance(in_biom, tuple):
        tables = [load_table(table) for table in in_biom]
    else:
        tables = [load_table(in_biom)]
    # import OrdinationResults
    ordination = OrdinationResults.read(in_ordination)
    # apply the transformation
    ord_res = _transform(ordination, tables,
                         subset_tables=subset_tables)
    # If it doesn't already exist, create the output directory.
    # Note that there is technically a race condition here: it's ostensibly
    # possible that some process could delete the output directory after we
    # check that it exists here but before we write the output files to it.
    # However, in this case, we'd just get an error from skbio.io.util.open()
    # (which is called by skbio.OrdinationResults.write()), which makes sense.
    os.makedirs(output_dir, exist_ok=True)
    # write results
    ord_res.write(os.path.join(output_dir, 'projected-ordination.txt'))


@cli.command(name='filter-ordination')
@click.option('--in-ordination',
              help=DESC_MORD,
              required=True)
@click.option('--in-biom',
              help=DESC_MTABLE,
              required=True)
@click.option('--output-dir',
              help='Location of output files.',
              required=True)
@click.option('--match-features',
              default=DEFAULT_MATCH,
              show_default=True,
              help=DESC_FM)
@click.option('--match-samples',
              default=DEFAULT_MATCH,
              show_default=True,
              help=DESC_SM)
def filter_ordination(in_ordination: str,
                      in_biom: str,
                      output_dir : str,
                      match_features : bool,
                      match_samples : bool) -> None:
    # import OrdinationResults
    ordination = OrdinationResults.read(in_ordination)
    # import table
    table = load_table(in_biom)
    out_ordination = _filter_ordination(ordination,
                                        table,
                                        match_features=match_features,
                                        match_samples=match_samples)
    # If it doesn't already exist, create the output directory.
    # Note that there is technically a race condition here: it's ostensibly
    # possible that some process could delete the output directory after we
    # check that it exists here but before we write the output files to it.
    # However, in this case, we'd just get an error from skbio.io.util.open()
    # (which is called by skbio.OrdinationResults.write()), which makes sense.
    os.makedirs(output_dir, exist_ok=True)
    # write results
    out_ordination.write(os.path.join(output_dir,
                                      'subset-ordination.txt'))


@cli.command(name='feature-correlation-table')
@click.option('--in-ordination',
              help=DESC_CORRTBLORD,
              required=True)
@click.option('--output-dir',
              help='Location of output files.',
              required=True)
def feature_correlation_table(in_ordination: str,
                              output_dir: str) -> None:
    # import OrdinationResults
    ordination = OrdinationResults.read(in_ordination)
    # import table
    corr_table = _feature_correlation_table(ordination)
    # If it doesn't already exist, create the output directory.
    # Note that there is technically a race condition here: it's ostensibly
    # possible that some process could delete the output directory after we
    # check that it exists here but before we write the output files to it.
    # However, in this case, we'd just get an error from skbio.io.util.open()
    # (which is called by skbio.OrdinationResults.write()), which makes sense.
    os.makedirs(output_dir, exist_ok=True)
    # write results
    out_ = os.path.join(output_dir, 'feature-correlation-table.tsv')
    corr_table.to_csv(out_, sep='\t')
