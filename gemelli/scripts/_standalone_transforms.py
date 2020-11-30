import os
import click
from .__init__ import cli
from biom import load_table
from biom.util import biom_open
from gemelli.preprocessing import rclr_transformation


@cli.command(name='rclr')
@click.option('--in-biom',
              help='Input table in biom format.',
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
