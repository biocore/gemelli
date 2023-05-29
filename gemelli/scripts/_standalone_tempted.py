import os
import click
from .__init__ import cli
import pandas as pd
from biom import load_table
from skbio import OrdinationResults
from gemelli.tempted import (tempted_factorize,
                             tempted_project)
from gemelli._defaults import (DEFAULT_COMP, 
                               DEFAULT_TEMPTED_EP,
                               DEFAULT_TEMPTED_SMTH,
                               DEFAULT_TEMPTED_RES,
                               DEFAULT_TEMPTED_MAXITER,
                               DEFAULT_TEMPTED_RHC,
                               DEFAULT_TEMPTED_SVDC,
                               DEFAULT_TEMPTED_SVDCN)
from gemelli._defaults import (DESC_COUNTS,
                               DESC_SMETA,
                               DESC_SUBJ,
                               DESC_TCOND,
                               DESC_OUT,
                               DESC_COMP,
                               DESC_REP,
                               DESC_SVD,
                               DESC_SVDC,
                               DESC_RES,
                               DESC_SMTH,
                               DESC_MXTR,
                               DESC_EPS,
                               DESC_IO,
                               DESC_SLO,
                               DESC_SVDO)


@cli.command(name='tempted')
@click.option('--in-biom',
              required=True,
              help=DESC_COUNTS)
@click.option('--sample-metadata-file',
              required=True,
              help=DESC_SMETA)
@click.option('--individual-id-column',
              required=True,
              help=DESC_SUBJ)
@click.option('--state-column',
              required=True,
              help=DESC_TCOND)
@click.option('--output-dir',
              required=True,
              help=DESC_OUT)
@click.option('--n-components',
              default=DEFAULT_COMP,
              show_default=True,
              help=DESC_COMP)
@click.option('--replicate-handling',
              default=DEFAULT_TEMPTED_RHC,
              show_default=True,
              help=DESC_REP)
@click.option('--svd-centralized',
              default=DEFAULT_TEMPTED_SVDC,
              show_default=True,
              help=DESC_SVD)
@click.option('--n-components-centralize',
              default=DEFAULT_TEMPTED_SVDCN,
              show_default=True,
              help=DESC_SVDC)
@click.option('--smooth',
              default=DEFAULT_TEMPTED_SMTH,
              show_default=True,
              help=DESC_SMTH)
@click.option('--resolution',
              default=DEFAULT_TEMPTED_RES,
              show_default=True,
              help=DESC_RES)
@click.option('--max-iterations',
              default=DEFAULT_TEMPTED_MAXITER,
              show_default=True,
              help=DESC_MXTR)
@click.option('--epsilon',
              default=DEFAULT_TEMPTED_EP,
              show_default=True,
              help=DESC_EPS)
def standalone_tempted(in_biom: str,
                       sample_metadata_file: str,
                       individual_id_column: str,
                       output_dir: str,
                       state_column: str,
                       n_components: int,
                       replicate_handling: str,
                       svd_centralized: bool,
                       n_components_centralize: int,
                       smooth: float,
                       resolution: int,
                       max_iterations: int,
                       epsilon: float) -> None:
    """Runs tempted on pre-transformed data."""
    # import table
    table = load_table(in_biom)
    # import sample metadata
    sample_metadata = pd.read_csv(sample_metadata_file,
                                  sep='\t', index_col=0,
                                  low_memory=False)
    # run tempted
    res_ = tempted_factorize(table,
                                  sample_metadata,
                                  individual_id_column,
                                  state_column,
                                  n_components=n_components,
                                  replicate_handling=replicate_handling,
                                  svd_centralized=svd_centralized,
                                  n_components_centralize=n_components_centralize,
                                  smooth=smooth,
                                  resolution=resolution,
                                  max_iterations=max_iterations,
                                  epsilon=epsilon)
    (individual_ord,
     state_loadings,
     dists,
     svd_center) = res_
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
    # write distance matrix
    dists.write(os.path.join(output_dir,
                             'individual-distance-matrix.txt'))
    # write each state ord
    individual_ord.write(os.path.join(output_dir,
                                      'individual-ordination.txt'))
    # write out dataframes
    state_loadings.to_csv(os.path.join(output_dir, 'state-loadings.tsv'),
                          sep='\t')
    svd_center.to_csv(os.path.join(output_dir, 'svd-center.tsv'),
                      sep='\t')


@cli.command(name='tempted-transform')
@click.option('--individual-ordination-file',
              required=True,
              help=DESC_IO)
@click.option('--state-loadings-file',
              required=True,
              help=DESC_SLO)
@click.option('--svd-center-file',
              required=True,
              help=DESC_SVDO)
@click.option('--in-biom',
              required=True,
              help=DESC_COUNTS)
@click.option('--sample-metadata-file',
              required=True,
              help=DESC_SMETA)
@click.option('--output-dir',
              required=True,
              help=DESC_OUT)
@click.option('--individual-id-column',
              required=True,
              help=DESC_SUBJ)
@click.option('--state-column',
              required=True,
              help=DESC_TCOND)
@click.option('--replicate-handling',
              default=DEFAULT_TEMPTED_RHC,
              show_default=True,
              help=DESC_REP)
def standalone_tempted_transform(individual_ordination_file: str,
                                 state_loadings_file: str,
                                 svd_center_file: str,
                                 in_biom: str,
                                 sample_metadata_file: str,
                                 output_dir: str,
                                 individual_id_column: str,
                                 state_column: str,
                                 replicate_handling: str) -> None:
    """Project new data into tempted."""
    # import table
    table = load_table(in_biom)
    # import sample metadata
    sample_metadata = pd.read_csv(sample_metadata_file,
                                  sep='\t', index_col=0,
                                  low_memory=False)
    # import sample metadata
    ind_ordination = OrdinationResults.read(individual_ordination_file)
    # import sample metadata
    state_loadings = pd.read_csv(state_loadings_file,
                                  sep='\t', index_col=0,
                                  low_memory=False)
    # import sample metadata
    svd_center = pd.read_csv(svd_center_file,
                                  sep='\t', index_col=0,
                                  low_memory=False)
    # project new data
    pord = tempted_project(ind_ordination,
                                            state_loadings,
                                            svd_center,
                                            table,
                                            sample_metadata,
                                            individual_id_column,
                                            state_column,
                                            replicate_handling)
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
    pord.write(os.path.join(output_dir, 'projected-ordination.txt'))
