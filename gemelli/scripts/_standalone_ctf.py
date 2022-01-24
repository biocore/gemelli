import os
import click
from .__init__ import cli
import pandas as pd
from biom.util import biom_open
from biom import load_table
from gemelli.ctf import (ctf_helper, phylogenetic_ctf_helper)
from gemelli.preprocessing import TaxonomyError
from gemelli._defaults import (DEFAULT_COMP, DEFAULT_MSC,
                               DEFAULT_MFC, DEFAULT_MTD,
                               DESC_MSC, DESC_MFC,
                               DEFAULT_TENSALS_MAXITER,
                               DEFAULT_FMETA, DESC_COMP,
                               DESC_OUT, DESC_ITERATIONSALS,
                               DESC_FMETA, DESC_COUNTS, DESC_SMETA,
                               DESC_SUBJ, DESC_COND, DESC_INIT,
                               DESC_MINDEPTH, DESC_TREE,
                               DEFAULT_MFF, DESC_MFF,
                               DESC_ITERATIONSRTPM, DEFAULT_COND)


@cli.command(name='phylogenetic-ctf')
@click.option('--in-biom',
              required=True,
              help=DESC_COUNTS)
@click.option('--in-phylogeny',
              help=DESC_TREE,
              required=True)
@click.option('--sample-metadata-file',
              required=True,
              help=DESC_SMETA)
@click.option('--individual-id-column',
              required=True,
              help=DESC_SUBJ)
@click.option('--state-column-1',
              required=True,
              help=DESC_COND)
@click.option('--output-dir',
              required=True,
              help=DESC_OUT)
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
@click.option('--max-iterations-als',
              default=DEFAULT_TENSALS_MAXITER,
              show_default=True,
              help=DESC_ITERATIONSALS)
@click.option('--max-iterations-rptm',
              default=DEFAULT_TENSALS_MAXITER,
              show_default=True,
              help=DESC_ITERATIONSRTPM)
@click.option('--n-initializations',
              default=DEFAULT_TENSALS_MAXITER,
              show_default=True,
              help=DESC_INIT)
@click.option('--feature-metadata-file',
              default=DEFAULT_FMETA,
              show_default=True,
              help=DESC_FMETA)
@click.option('--state-column-2',
              required=False,
              default=DEFAULT_COND,
              help=DESC_COND)
@click.option('--state-column-3',
              required=False,
              default=DEFAULT_COND,
              help=DESC_COND)
@click.option('--state-column-4',
              required=False,
              default=DEFAULT_COND,
              help=DESC_COND)
def standalone_phylogenetic_ctf(in_biom: str,
                                in_phylogeny: str,
                                sample_metadata_file: str,
                                individual_id_column: str,
                                state_column_1: str,
                                output_dir: str,
                                n_components: int,
                                min_sample_count: int,
                                min_feature_count: int,
                                min_feature_frequency: float,
                                min_depth: int,
                                max_iterations_als: int,
                                max_iterations_rptm: int,
                                n_initializations: int,
                                feature_metadata_file: None,
                                state_column_2: str,
                                state_column_3: str,
                                state_column_4: str) -> None:
    """Runs phylogenetically informed CTF with an rclr preprocessing step."""

    # generate state lists
    state_columns = [state for state in [state_column_1, state_column_2,
                                         state_column_3, state_column_4]
                     if state is not None]
    # import table
    table = load_table(in_biom)
    # import sample metadata
    sample_metadata = pd.read_csv(sample_metadata_file,
                                  sep='\t', index_col=0,
                                  low_memory=False)
    # import feature metadata if available
    feature_metadata = None
    if feature_metadata_file is not None:
        feature_metadata = pd.read_csv(feature_metadata_file,
                                       sep='\t', index_col=0,
                                       low_memory=False)
        try:
            taxonomy_table.set_index('Feature ID', inplace=True)
        except KeyError:
            raise TaxonomyError(
                        "Taxonomy file must have a column labled 'Feature ID'."
                        )
    # run CTF
    res_ = phylogenetic_ctf_helper(table,
                                   in_phylogeny,
                                   sample_metadata,
                                   individual_id_column,
                                   state_columns,
                                   n_components,
                                   min_sample_count,
                                   min_feature_count,
                                   min_feature_frequency,
                                   min_depth,
                                   max_iterations_als,
                                   max_iterations_rptm,
                                   n_initializations,
                                   feature_metadata)
    (state_ordn, ord_res, dists, straj, ftraj,
     phylogeny, counts_by_node,
     result_taxonomy, subject_table) = res_
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
    if result_taxonomy is not None:
        result_taxonomy.to_csv(os.path.join(output_dir, 't2t-taxonomy.tsv'),
                           sep='\t')
    ord_res.write(os.path.join(output_dir, 'ordination.txt'))
    # write each distance matrix
    for condition, dist in dists.items():
        dist.write(os.path.join(output_dir,
                                '%s-distance-matrix.tsv' % (str(condition))))
    # write each state ord
    for condition, ord_ in state_ordn.items():
        ord_.write(os.path.join(output_dir, '%s-ordination.txt' % (condition)))
    # write each trajectory
    for condition, traj in straj.items():
        traj.to_csv(
            os.path.join(
                output_dir,
                '%s-subject-ordination.tsv' %
                (str(condition))),
            sep='\t')
    # write each trajectory
    for condition, traj in ftraj.items():
        traj.to_csv(
            os.path.join(
                output_dir,
                '%s-features-ordination.tsv' %
                (str(condition))),
            sep='\t')
    # export phylogeny
    phylogeny.write(os.path.join(output_dir, 'labeled-phylogeny.nwk'))
    # write the vectorized count table for Qurro / log-ratios
    table_out_path = os.path.join(output_dir,
                                  'phylogenetic-table.biom')
    with biom_open(table_out_path, 'w') as f:
        counts_by_node.to_hdf5(f, "phylogenetic-ctf-count-table")
    table_out_path = os.path.join(output_dir,
                                  'subject-phylogenetic-table.biom')
    with biom_open(table_out_path, 'w') as f:
        subject_table.to_hdf5(f, "subject-ctf-count-table")


@cli.command(name='ctf')
@click.option(
    '--in-biom',
    required=True,
    help=DESC_COUNTS)
@click.option(
    '--sample-metadata-file',
    required=True,
    help=DESC_SMETA)
@click.option(
    '--individual-id-column',
    required=True,
    help=DESC_SUBJ)
@click.option(
    '--state-column-1',
    required=True,
    help=DESC_COND)
@click.option(
    '--output-dir',
    required=True,
    help=DESC_OUT)
@click.option(
    '--n-components',
    default=DEFAULT_COMP,
    show_default=True,
    help=DESC_COMP)
@click.option(
    '--min-sample-count',
    default=DEFAULT_MSC,
    show_default=True,
    help=DESC_MSC)
@click.option(
    '--min-feature-count',
    default=DEFAULT_MFC,
    show_default=True,
    help=DESC_MFC)
@click.option(
    '--min-feature-frequency',
    default=DEFAULT_MFF,
    show_default=True,
    help=DESC_MFF)
@click.option(
    '--max-iterations-als',
    default=DEFAULT_TENSALS_MAXITER,
    show_default=True,
    help=DESC_ITERATIONSALS)
@click.option(
    '--max-iterations-rptm',
    default=DEFAULT_TENSALS_MAXITER,
    show_default=True,
    help=DESC_ITERATIONSRTPM)
@click.option(
    '--n-initializations',
    default=DEFAULT_TENSALS_MAXITER,
    show_default=True,
    help=DESC_INIT)
@click.option(
    '--feature-metadata-file',
    default=DEFAULT_FMETA,
    show_default=True,
    help=DESC_FMETA)
@click.option(
    '--state-column-2',
    required=False,
    default=DEFAULT_COND,
    help=DESC_COND)
@click.option(
    '--state-column-3',
    required=False,
    default=DEFAULT_COND,
    help=DESC_COND)
@click.option(
    '--state-column-4',
    required=False,
    default=DEFAULT_COND,
    help=DESC_COND)
def standalone_ctf(in_biom: str,
                   sample_metadata_file: str,
                   individual_id_column: str,
                   state_column_1: str,
                   output_dir: str,
                   n_components: int,
                   min_sample_count: int,
                   min_feature_count: int,
                   min_feature_frequency: float,
                   max_iterations_als: int,
                   max_iterations_rptm: int,
                   n_initializations: int,
                   feature_metadata_file: str,
                   state_column_2: str,
                   state_column_3: str,
                   state_column_4: str) -> None:
    """Runs CTF with an rclr preprocessing step."""

    # generate state lists
    state_columns = [state for state in [state_column_1, state_column_2,
                                         state_column_3, state_column_4]
                     if state is not None]
    # import table
    table = load_table(in_biom)
    # import sample metadata
    sample_metadata = pd.read_csv(sample_metadata_file,
                                  sep='\t', index_col=0,
                                  low_memory=False)
    # import feature metadata if available
    if feature_metadata_file is not None:
        feature_metadata = pd.read_csv(feature_metadata_file,
                                       sep='\t', index_col=0,
                                       low_memory=False)
    else:
        feature_metadata = None
    # run CTF
    res_ = ctf_helper(table,
                      sample_metadata,
                      individual_id_column,
                      state_columns,
                      n_components,
                      min_sample_count,
                      min_feature_count,
                      min_feature_frequency,
                      max_iterations_als,
                      max_iterations_rptm,
                      n_initializations,
                      feature_metadata)
    state_ordn, ord_res, dists, straj, ftraj = res_
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
    # write each distance matrix
    for condition, dist in dists.items():
        dist.write(os.path.join(output_dir,
                                '%s-distance-matrix.tsv' % (str(condition))))
    # write each state ord
    for condition, ord_ in state_ordn.items():
        ord_.write(os.path.join(output_dir, '%s-ordination.txt' % (condition)))
    # write each trajectory
    for condition, traj in straj.items():
        traj.to_csv(
            os.path.join(
                output_dir,
                '%s-subject-ordination.tsv' %
                (str(condition))),
            sep='\t')
    # write each trajectory
    for condition, traj in ftraj.items():
        traj.to_csv(
            os.path.join(
                output_dir,
                '%s-features-ordination.tsv' %
                (str(condition))),
            sep='\t')
