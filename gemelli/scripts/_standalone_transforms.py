import os
import click
from .__init__ import cli
from biom import load_table
from biom.util import biom_open
from gemelli.preprocessing import (rclr_transformation,
                                   phylogenetic_rclr_transformation,
                                   clr_transformation,
                                   phylogenetic_clr_transformation)
from gemelli._defaults import (DESC_COUNTS, DESC_TREE, DESC_PC)


@cli.command(name='phylogenetic-rclr')
@click.option('--in-biom',
              help=DESC_COUNTS,
              required=True)
@click.option('--in-phylogeny',
              help=DESC_TREE,
              required=True)
@click.option('--output-dir',
              help='Location of output table.',
              required=True)
def standalone_phylogenetic_rclr(in_biom: str,
                                 in_phylogeny: str,
                                 output_dir: str) -> None:
    """
    Runs phylogenetic robust centered log-ratio transformation.
    Returns both a vectorized table and matched fully labeled phylogeny
    in addition to a rclr transformed version of the phylogenetic table.
    Note: This is run automatically within phylo-CTF/RPCA
    so there no need to run rclr before those functions.
    """

    # import table
    table = load_table(in_biom)
    # run vectorized table and rclr transform
    res_ = phylogenetic_rclr_transformation(table, in_phylogeny)
    counts_by_node, rclr_table, phylogeny = res_

    # If it doesn't already exist, create the output directory.
    # Note that there is technically a race condition here: it's ostensibly
    # possible that some process could delete the output directory after we
    # check that it exists here but before we write the output files to it.
    # However, in this case, we'd just get an error from skbio.io.util.open()
    # (which is called by skbio.OrdinationResults.write()), which makes sense.
    os.makedirs(output_dir, exist_ok=True)

    # write files to output directory
    phylogeny.write(os.path.join(output_dir, 'labeled-phylogeny.nwk'))
    out_path = os.path.join(output_dir, 'phylogenetic-rclr-table.biom')
    with biom_open(out_path, 'w') as wf:
        rclr_table.to_hdf5(wf, "phylogenetic-rclr-table")
    out_path = os.path.join(output_dir, 'phylogenetic-count-table.biom')
    with biom_open(out_path, 'w') as wf:
        counts_by_node.to_hdf5(wf, "phylogenetic-table")


@cli.command(name='rclr')
@click.option('--in-biom',
              help=DESC_COUNTS,
              required=True)
@click.option('--output-dir',
              help='Location of output table.',
              required=True)
def standalone_rclr(in_biom: str,
                    output_dir: str) -> None:
    """
    Runs robust centered log-ratio transformation.
    Note: This is run automatically within CTF/RPCA/Auto-RPCA
    so there no need to run rclr before those functions.
    """

    # import table and perform rclr transform
    table = rclr_transformation(load_table(in_biom))

    # If it doesn't already exist, create the output directory.
    # Note that there is technically a race condition here: it's ostensibly
    # possible that some process could delete the output directory after we
    # check that it exists here but before we write the output files to it.
    # However, in this case, we'd just get an error from skbio.io.util.open()
    # (which is called by skbio.OrdinationResults.write()), which makes sense.
    os.makedirs(output_dir, exist_ok=True)

    # write files to output directory
    out_path = os.path.join(output_dir, 'rclr-table.biom')
    with biom_open(out_path, 'w') as wf:
        table.to_hdf5(wf, "rclr-table")


@cli.command(name='phylogenetic-clr')
@click.option('--in-biom',
              help=DESC_COUNTS,
              required=True)
@click.option('--in-phylogeny',
              help=DESC_TREE,
              required=True)
@click.option('--output-dir',
              help='Location of output table.',
              required=True)
@click.option('--pseudocount',
              default=0.0,
              show_default=True,
              help=DESC_PC)
def standalone_phylogenetic_clr(in_biom: str,
                                in_phylogeny: str,
                                output_dir: str,
                                pseudocount: float) -> None:
    """
    Runs phylogenetic robust centered log-ratio transformation.
    Returns both a vectorized table and matched fully labeled phylogeny
    in addition to a clr transformed version of the phylogenetic table.
    By default a pseudocount is added with the minimum non-zero value.
    """

    # import table
    table = load_table(in_biom)
    # run vectorized table and clr transform
    res_ = phylogenetic_clr_transformation(table, in_phylogeny,
                                           pseudocount=pseudocount)
    counts_by_node, clr_table, phylogeny = res_

    # If it doesn't already exist, create the output directory.
    # Note that there is technically a race condition here: it's ostensibly
    # possible that some process could delete the output directory after we
    # check that it exists here but before we write the output files to it.
    # However, in this case, we'd just get an error from skbio.io.util.open()
    # (which is called by skbio.OrdinationResults.write()), which makes sense.
    os.makedirs(output_dir, exist_ok=True)

    # write files to output directory
    phylogeny.write(os.path.join(output_dir, 'labeled-phylogeny.nwk'))
    out_path = os.path.join(output_dir, 'phylogenetic-clr-table.biom')
    with biom_open(out_path, 'w') as wf:
        clr_table.to_hdf5(wf, "phylogenetic-clr-table")
    out_path = os.path.join(output_dir, 'phylogenetic-count-table.biom')
    with biom_open(out_path, 'w') as wf:
        counts_by_node.to_hdf5(wf, "phylogenetic-table")


@cli.command(name='clr')
@click.option('--in-biom',
              help=DESC_COUNTS,
              required=True)
@click.option('--output-dir',
              help='Location of output table.',
              required=True)
@click.option('--pseudocount',
              default=0.0,
              show_default=True,
              help=DESC_PC)
def standalone_clr(in_biom: str,
                   output_dir: str,
                   pseudocount: float) -> None:
    """
    Runs robust centered log-ratio transformation.
    By default a pseudocount is added with the
    minimum non-zero value.
    """

    # import table and perform clr transform
    table = clr_transformation(load_table(in_biom),
                               pseudocount=pseudocount)

    # If it doesn't already exist, create the output directory.
    # Note that there is technically a race condition here: it's ostensibly
    # possible that some process could delete the output directory after we
    # check that it exists here but before we write the output files to it.
    # However, in this case, we'd just get an error from skbio.io.util.open()
    # (which is called by skbio.OrdinationResults.write()), which makes sense.
    os.makedirs(output_dir, exist_ok=True)

    # write files to output directory
    out_path = os.path.join(output_dir, 'clr-table.biom')
    with biom_open(out_path, 'w') as wf:
        table.to_hdf5(wf, "clr-table")
