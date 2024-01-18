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
from gemelli.utils import qc_rarefaction as _qc_rarefaction
import pkg_resources
import seaborn as sns
import itertools
import matplotlib.pyplot as plt

TEMPLATES = pkg_resources.resource_filename('gemelli', 'q2')


def qc_rarefy(output_dir: str,
              table: biom.Table,
              rarefied_distance: DistanceMatrix,
              unrarefied_distance: DistanceMatrix,
              permutations: int = 999) -> None:
    """
    Much of this code copies the method used in:
    qiime2/q2-diversity/q2_diversity/_beta/_visualizer.py
    for the visualization of a correlation. It has been
    adapted for the viusalization of the distance qc
    metric from Schloss, 2023 to QC the distances.
    https://doi.org/10.1101/2023.06.23.546313.
    """
    # run mantel test and get stats on comparison
    (t_, p_, 
     xy, p_xy, 
     xz, pxz_, yz,
     samp_sum_dist) = _qc_rarefaction(table,
                                      rarefied_distance, 
                                      unrarefied_distance,
                                      permutations,
                                      True)
    # build visualization table
    sample_size = len(samp_sum_dist.ids)
    result = pd.Series(['Steiger', sample_size,
                        'two-sided', t_, p_],
                       index=['Method', 'Sample size',
                              'Alternative hypothesis',
                              'test statistic',
                              'p-value'],
                       name='Steiger test results between rarefied'
                            ' and unrarefied distances.')
    table_html = q2templates.df_to_html(result.to_frame())
    # build visualization plots
    plot_data_iter = [[rarefied_distance, samp_sum_dist, xy, p_xy,
                       'Rarefied Distance vs. Sample Abs. Sum Differences'],
                       [rarefied_distance, samp_sum_dist, xz, pxz_,
                       'Unrarefied Distance vs. Sample Abs. Sum Differences']]
    label1 = 'Distances'
    label2 = 'Abs. Sample Sum Differences'
    for p_i, (dm1, dm2, r_m, p_m, title_) in enumerate(plot_data_iter):
        # build visuals
        # We know the distance matrices have matching ID sets at this point, so we
        # can safely generate all pairs of IDs using one of the matrices' ID sets
        # (it doesn't matter which one).
        scatter_data = []
        for id1, id2 in itertools.combinations(dm1.ids, 2):
            scatter_data.append((dm1[id1, id2], dm2[id1, id2]))
        plt.figure()
        x = '%s' % label1
        y = '%s' % label2
        scatter_data = pd.DataFrame(scatter_data, columns=[x, y])
        sns.regplot(x=x, y=y, data=scatter_data, fit_reg=False)
        title_ = title_ + '\n(r=%.3f, p=%.3f)' % (r_m, p_m)
        plt.title(title_, fontsize=12, color='black')
        plt.savefig(os.path.join(output_dir, 'mantel-scatter-%i.svg' % (p_i + 1)))
        plt.close()
    # buld final visual
    context = {
        'table': table_html,
    }
    index = os.path.join(
        TEMPLATES, 'qc_assests', 'index.html')
    q2templates.render(index, output_dir, context=context)
