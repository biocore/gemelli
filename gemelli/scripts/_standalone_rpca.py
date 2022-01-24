import os
import click
import pandas as pd
from .__init__ import cli
from biom import load_table
from biom.util import biom_open
from gemelli.preprocessing import TaxonomyError
from gemelli.rpca import rpca as _rpca
from gemelli.rpca import auto_rpca as _auto_rpca
from gemelli.rpca import phylogenetic_rpca as _phylo_rpca
from gemelli._defaults import (DEFAULT_COMP, DEFAULT_MSC, DEFAULT_MTD,
                               DEFAULT_MFC, DEFAULT_OPTSPACE_ITERATIONS,
                               DESC_COMP, DESC_MSC, DESC_MFC,
                               DESC_ITERATIONS, DESC_MINDEPTH,
                               DEFAULT_MFF, DESC_MFF,
                               DESC_COUNTS, DESC_TREE, DESC_TAX_SA)


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


@cli.command(name='auto-rpca')
@click.option('--in-biom',
              help=DESC_COUNTS,
              required=True)
@click.option('--output-dir',
              help='Location of output files.',
              required=True)
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
    '--max-iterations',
    default=DEFAULT_OPTSPACE_ITERATIONS,
    show_default=True,
    help=DESC_ITERATIONS)
def auto_rpca(in_biom: str,
              output_dir: str,
              min_sample_count: int,
              min_feature_count: int,
              min_feature_frequency: float,
              max_iterations: int) -> None:
    """Runs RPCA with an rclr preprocessing step and auto-estimates the
       rank (i.e. n-components parameter)."""

    # import table
    table = load_table(in_biom)
    # run the RPCA wrapper
    ord_res, dist_res = _auto_rpca(table,
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
