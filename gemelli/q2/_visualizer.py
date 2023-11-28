import biom
import scipy
import os
import shutil
import json
import qiime2 as q2
import q2templates
import pandas as pd
from urllib.parse import quote
from skbio import DistanceMatrix
from gemelli.utils import qc_distances as _qc_distances
import pkg_resources

TEMPLATES = pkg_resources.resource_filename('q2_diversity', '_beta')

def qc_distances(ctx, distance, table,
                 method='spearman',
                 permutations=999):
    """
    Much of this code copies the method used in:
    qiime2/q2-diversity/q2_diversity/_beta/_visualizer.py
    for the visualization of a correlation. It has been
    adapted for the viusalization of the distance qc
    metric from Schloss, 2023 to QC the distances.
    https://doi.org/10.1101/2023.06.23.546313.
    """

    results = []
    # generate matched "distances" to compare
    dm1, dm2 = _qc_distances(distance.view(DistanceMatrix),
                             table.view(biom.Table))
    dm1 = q2.Artifact.import_data('DistanceMatrix', dm1)
    dm2 = q2.Artifact.import_data('DistanceMatrix', dm2)
    results.append(dm1)
    results.append(dm2)
    label1 = 'Distances'
    label2 = 'Abs. Sample Sum Differences'
    # run mantel corr.
    mantel = ctx.get_action('diversity', 'mantel')
    mantel_visualization, = mantel(dm1, dm2,
                                   method, permutations, 
                                   False, label1, label2)
    results.append(mantel_visualization)
    return tuple(results)
