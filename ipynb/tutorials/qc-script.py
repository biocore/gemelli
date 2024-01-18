from biom import load_table
import pandas as pd
import numpy as np
from skbio.diversity import beta_diversity
from gemelli.rpca import rpca
from gemelli.utils import (qc_rarefaction, qc_distances)
from tqdm import tqdm
np.seterr(divide = 'ignore') 

"""
Script for testing mass amounts of studies downloaded
bulk (see above) w/ qc_rarefaction for RPCA 
for introduction tutorial.
"""
# minimum sample size to keep after filter/subset
min_sample_size_cutoff = 10
# number of iterations in rarefaction
n_iter_subsample = 100
# table filters
# these are a little strict
# but we are blindly running a bunch of very 
# different types of studies. 
# In practice, they could probably be
# optimized/decreased for each study individually.
# It also helps make the run time reasonable since
# we are doing 100 subsamples.
min_feature_count = 2
min_feature_frequency = 10
min_sample_count = 10000
# import data (all Qiita fecal)
bulk_studies_mf = pd.read_csv('bulk-qiita.txt', sep='\t', index_col=0)
bulk_studies_mf['qiita_study_id'] =  [int(x.split('.')[0]) for x in bulk_studies_mf.index]
bulk_studies_bt = load_table('bulk-qiita.biom')
# Double check no repeated measure subjects 
# (should be done with repeated measure method)
# otherwise seq. depth will be non-randomly associated to subject.
bulk_studies_mf_hscount = bulk_studies_mf.host_subject_id.value_counts()
bulk_studies_mf = bulk_studies_mf[(~bulk_studies_mf.host_subject_id.isin(bulk_studies_mf_hscount[bulk_studies_mf_hscount > 1].index))]
# for each study run RPCA test 
qc_results_testing = {}
bc_qc_results_testing = {}
# try previous
for study_, study_mf_ in tqdm(bulk_studies_mf.groupby('qiita_study_id')):
    # drop any blanks/control/mock samples
    study_mf_ = study_mf_.drop([x for x in study_mf_.index if 'blank' in x.lower()])
    study_mf_ = study_mf_.drop([x for x in study_mf_.index if 'control' in x.lower()])
    study_mf_ = study_mf_.drop([x for x in study_mf_.index if 'mock' in x.lower()])
    study_mf_ = study_mf_.drop([x for x in study_mf_.index if 'zymo' in x.lower()])
    n_ = study_mf_.shape[0]
    if n_ <= min_sample_size_cutoff:
        continue
    # match 
    shared_ = set(bulk_studies_bt.ids()) & set(study_mf_.index)
    bt_study_ = bulk_studies_bt.copy().filter(shared_)
    #bt_study_ = bt_study_.filter(sample_filter, axis='sample')
    n_features, n_samples = bt_study_.shape
    bt_study_ = bt_study_.filter(bt_study_.ids('observation')[bt_study_.sum('observation') > min_feature_count], axis='observation')
    bt_study_ = bt_study_.filter(bt_study_.ids('observation')[(np.array(bt_study_.matrix_data.astype(bool).sum(1)).ravel().ravel() / n_samples) > (min_feature_frequency/100)], axis='observation')
    bt_study_ = bt_study_.filter(bt_study_.ids()[bt_study_.sum('sample') >= min_sample_count])
    # either the sample cutoff or min. sample after cutoff
    n_ = int(len(bt_study_.ids()))
    if n_ <= min_sample_size_cutoff:
        continue
    rare_level =  int(max(int(min_sample_count),
                          int(bt_study_.sum('sample').min())))
    # non-rare run before
    ord_norare, dist_norare = rpca(bt_study_)
    # BC non-rare run before
    bc_dist_norare = beta_diversity('braycurtis', bt_study_.matrix_data.toarray().T, ids=bt_study_.ids())
    # get the sample sum dist
    (_, samp_sum_dist) = qc_distances(dist_norare,
                                      bt_study_)
    for fold_rare in range(n_iter_subsample):
        # run RPCA w/o CV
        # (just interested in rare and don't want to introduce artifacts from randomly chosen projections)
        ord_rare, dist_rare = rpca(bt_study_.copy().subsample(rare_level)) 
        # run QC (for runtime sake we will just calculate the correlations we need for mantel not the p-values)
        t_, p_, xy, p_xy, xz, pxz_, yz, samp_sum_dist = qc_rarefaction(bt_study_, 
                                                                       dist_rare, 
                                                                       dist_norare, 
                                                                       mantel_permutations=1,
                                                                       return_mantel=True,
                                                                       samp_sum_dist=samp_sum_dist)
        qc_results_testing[(study_, fold_rare)] = [t_, p_, xy, xz, yz]
        # BC rare
        bc_dist_rare = beta_diversity('braycurtis', bt_study_.copy().subsample(rare_level).matrix_data.toarray().T, ids=bt_study_.ids())
        # BC run QC (for runtime sake we will just calculate the correlations we need for mantel not the p-values)
        bct_, bcp_, bcxy, bcp_xy, bcxz, bcpxz_, bcyz, bcsamp_sum_dist = qc_rarefaction(bt_study_, 
                                                                                       bc_dist_rare, 
                                                                                       bc_dist_norare, 
                                                                                       mantel_permutations=1,
                                                                                       return_mantel=True,
                                                                                       samp_sum_dist=samp_sum_dist)
        bc_qc_results_testing[(study_, fold_rare)] = [bct_, bcp_, bcxy, bcxz, bcyz]

# conbine and save as DF
columns_ = ['t', 'p',
            'rare_mantel_r',
            'unrare_mantel_r',
            'rare_unrare_mantel_r']
# RPCA results
qc_results_testing_df = pd.DataFrame(qc_results_testing, columns_).T
qc_results_testing_df = qc_results_testing_df.reset_index().rename({'level_0':'study_id',
                                                                    'level_1':'fold_rare',}, axis=1)
qc_results_testing_df.to_csv('qc-rare-results-all-test-10.csv')
# BC results
bc_qc_results_testing_df = pd.DataFrame(bc_qc_results_testing, columns_).T
bc_qc_results_testing_df = bc_qc_results_testing_df.reset_index().rename({'level_0':'study_id',
                                                                    'level_1':'fold_rare',}, axis=1)
bc_qc_results_testing_df.to_csv('bc-qc-rare-results-all-test-10.csv')
