"""
In order to download the raw-data use the following script (run here on 12/01/2023)
# set context for data type (just do 16S for now)
export ctx=Deblur_2021.09-Illumina-16S-V4-100nt-50b3a2
# all EMP (environmental)
redbiom search metadata EMP | redbiom fetch samples --context $ctx --output bulk-qiita-emp.biom
redbiom search metadata EMP | redbiom fetch sample-metadata --context $ctx --output bulk-qiita-emp.txt
# all feces
redbiom search metadata feces | redbiom fetch samples --context $ctx --output bulk-qiita-feces.biom
redbiom search metadata feces | redbiom fetch sample-metadata --context $ctx --output bulk-qiita-feces.txt --all-columns
""" 

from biom import load_table
import pandas as pd
import numpy as np
from gemelli.rpca import rpca
from gemelli.utils import (qc_rarefaction, qc_distances)

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
min_feature_frequency = 25
min_sample_count = 10000
# import data (all Qiita fecal)
bulk_studies_mf = pd.read_csv('bulk-qiita-feces.txt', sep='\t', index_col=0)
bulk_studies_mf['qiita_study_id'] =  [int(x.split('.')[0]) for x in bulk_studies_mf.index]
bulk_studies_bt = load_table('bulk-qiita-feces.biom')
# Drop repeated measure subjects 
# (should be done with repeated measure method)
# otherwise seq. depth will be non-randomly associated
# to subject.
bulk_studies_mf_hscount = bulk_studies_mf.host_subject_id.value_counts()
bulk_studies_mf = bulk_studies_mf[~bulk_studies_mf.host_subject_id.isin(bulk_studies_mf_hscount[bulk_studies_mf_hscount > 1].index)]
# double check studys with _any_ easily identifiable repeated measures are gone
# should go back at some point and check with CTF/TEMPTED but
# that will require a study by study analysis since we need
# to understand how the samples distribute across repeated measures.
drop_ = [1288, 11405, 10894, 1038, 2192, 2318, 11666, 10394, 2202, 11166, 
         11937, 10533, 678, 1191, 2086, 10793, 1579, 11052, 10156, 940,
         10925, 945, 1718, 10171, 11710, 10689, 10180, 10184, 13512, 10317, 
         2382, 1998, 12496, 723, 1622, 11479, 1240, 
         11358, 990, 864, 1634, 11874, 2538, 1642, 11884, 
         11757, 11882, 755, 894]
# ^ there are a lot more than I realized
bulk_studies_mf = bulk_studies_mf[~bulk_studies_mf.qiita_study_id.isin(drop_)]
# for each study run RPCA test 
qc_results_testing_fecal = {}
# try previous
for study_, study_mf_ in bulk_studies_mf.groupby('qiita_study_id'):
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
        qc_results_testing_fecal[(study_, fold_rare)] = [t_, p_, xy, p_xy, xz, pxz_, yz]
# import data (EMP)
bulk_studies_mf = pd.read_csv('bulk-studies-emp.txt', sep='\t', index_col=0)
bulk_studies_bt = load_table('bulk-qiita-emp.biom')
# Drop repeated measure subjects 
# (should be done with repeated measure method)
# otherwise seq. depth will be non-randomly associated
# to subject. This approach is a little naive, would
# be better to go through by hand study by study but
# this will do for now.
bulk_studies_mf_hscount = bulk_studies_mf.host_subject_id.value_counts()
bulk_studies_mf = bulk_studies_mf[~bulk_studies_mf.host_subject_id.isin(bulk_studies_mf_hscount[bulk_studies_mf_hscount > 1].index)]
# double check easily identifiable repeated measure studys are gone
bulk_studies_mf = bulk_studies_mf[~bulk_studies_mf.qiita_study_id.isin(drop_)]
# for each study run RPCA test 
qc_results_testing_emp = {}
# try previous
for study_, study_mf_ in bulk_studies_mf.groupby('qiita_study_id'):
    # drop any blanks/control/mock samples
    study_mf_ = study_mf_.drop([x for x in study_mf_.index if 'blank' in x.lower()])
    study_mf_ = study_mf_.drop([x for x in study_mf_.index if 'control' in x.lower()])
    study_mf_ = study_mf_.drop([x for x in study_mf_.index if 'mock' in x.lower()])
    n_ = study_mf_.shape[0]
    if n_ <= min_sample_size_cutoff:
        continue
    study_mf_ = study_mf_[study_mf_.description != 'Control']
    study_mf_ = study_mf_[['control' not in x.lower() for x in study_mf_.description]]
    study_mf_ = study_mf_[['blank' not in x.lower()  for x in study_mf_.description]]
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
        qc_results_testing_emp[(study_, fold_rare)] = [t_, p_, xy, p_xy, xz, pxz_, yz]

# conbine and save as DF
columns_ = ['t', 'p',
            'rare_mantel_r', 'rare_mantel_p',  
            'unrare_mantel_r', 'unrare_mantel_p',
            'rare_unrare_mantel_r']
qc_results_testing_df = pd.DataFrame({**qc_results_testing_fecal, **qc_results_testing_emp}, columns_).T
qc_results_testing_df = qc_results_testing_df.reset_index().rename({'level_0':'study_id',
                                                                    'level_1':'fold_rare',}, axis=1)
qc_results_testing_df.to_csv('qc-rare-results-all.csv')
