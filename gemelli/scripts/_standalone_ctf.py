import os
import click
from .__init__ import cli
import pandas as pd
from biom import load_table
from gemelli.ctf import (ctf_helper, phylogenetic_ctf_helper)
from gemelli._defaults import (DEFAULT_COMP, DEFAULT_MSC,
                               DEFAULT_MFC, DEFAULT_MTD,
                               DESC_MSC, DESC_MFC,
                               DEFAULT_TENSALS_MAXITER,
                               DEFAULT_FMETA, DESC_COMP,
                               DESC_OUT, DESC_ITERATIONSALS,
                               DESC_FMETA, DESC_COUNTS, DESC_SMETA,
                               DESC_SUBJ, DESC_COND, DESC_INIT,
                               DESC_MINDEPTH, DESC_MINSPLIT,
                               DESC_MINPOST, DESC_TREE,
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
@click.option('--n_components',
              default=DEFAULT_COMP,
              show_default=True,
              help=DESC_COMP)
@click.option('--min-sample-count',
              default=DEFAULT_MSC,
              show_default=True,
              help=DESC_MSC)
@click.option('--min_depth',
              default=DEFAULT_MTD,
              show_default=True,
              help=DESC_MINDEPTH)
@click.option('--min_splits',
              default=DEFAULT_MTD,
              show_default=True,
              help=DESC_MINSPLIT)
@click.option('--max_postlevel',
              default=DEFAULT_MTD,
              show_default=True,
              help=DESC_MINPOST)
@click.option('--min-feature-count',
              default=DEFAULT_MFC,
              show_default=True,
              help=DESC_MFC)
@click.option('--max_iterations_als',
              default=DEFAULT_TENSALS_MAXITER,
              show_default=True,
              help=DESC_ITERATIONSALS)
@click.option('--max_iterations_rptm',
              default=DEFAULT_TENSALS_MAXITER,
              show_default=True,
              help=DESC_ITERATIONSRTPM)
@click.option('--n_initializations',
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
                   in_tree: str,
                   sample_metadata_file: str,
                   individual_id_column: str,
                   state_column_1: str,
                   output_dir: str,
                   n_components: int,
                   min_sample_count: int,
                   min_feature_count: int,
                   min_depth: int,
                   min_splits: int,
                   max_postlevel: int,
                   max_iterations_als: int,
                   max_iterations_rptm: int,
                   n_initializations: int,
                   feature_metadata_file: str,
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
    # import phylogeny
    with in_tree.open() as fh:
        phylogeny = TreeNode.read(fh, format='newick')
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
    res_ = phylogenetic_ctf_helper(table,
                                   phylogeny,
                                   sample_metadata,
                                   individual_id_column,
                                   state_columns,
                                   n_components,
                                   min_sample_count,
                                   min_feature_count,
                                   min_depth,
                                   min_splits,
                                   max_postlevel,
                                   max_iterations_als,
                                   max_iterations_rptm,
                                   n_initializations,
                                   feature_metadata)
    state_ordn, ord_res, dists, straj, ftraj, phylogeny, counts_by_node  = res_
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
    # export phylogeny
    phylogeny.write(os.path.join(output_dir, 'labeled-phylogeny.tsv'))
    # write the vectorized count table for Qurro / log-ratios
    with biom_open(os.path.join(output_dir, 'phylo-table.biom'), 'w') as f:
        counts_by_node.to_hdf5(f, "phylo-rpca-count-table")


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
    '--n_components',
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
    '--max_iterations_als',
    default=DEFAULT_TENSALS_MAXITER,
    show_default=True,
    help=DESC_ITERATIONSALS)
@click.option(
    '--max_iterations_rptm',
    default=DEFAULT_TENSALS_MAXITER,
    show_default=True,
    help=DESC_ITERATIONSRTPM)
@click.option(
    '--n_initializations',
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
