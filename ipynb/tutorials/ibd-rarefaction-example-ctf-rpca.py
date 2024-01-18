from gemelli.rpca import rpca_table_processing, rpca
from gemelli.ctf import ctf
from gemelli.utils import (qc_rarefaction, qc_distances)
from tqdm import tqdm
import pandas as pd
from biom import load_table

def rarefaction_compare_ibd_w_ctf(bt,  mf, taxonomy,
                                  rarefy_level, 
                                  n_rarefy=100):
    """
    Quick helper function to compare QC function for
    repeated measure example.
    
    bt: biom.Table
    Table to run.
    
    n_rarefy: int
    Number of time to rarefy.
    """
    test_results = {}
    for i in tqdm(range(n_rarefy)):
        # RPCA
        # run unrare data
        ord_norare, dist_norare = rpca(bt,
                                       min_sample_count=rarefy_level)   
        # rarefy data and rerun
        rare_bt = bt.copy().subsample(rarefy_level)
        ord_rare, dist_rare = rpca(rare_bt)
        # qc the data
        t_, p_, xy, _, xz, _, _, _ = qc_rarefaction(bt, dist_rare, 
                                                    dist_norare, 
                                                    return_mantel=True,
                                                    mantel_permutations=1)
        # save the results
        test_results[(i, 'RPCA')] = [t_, p_, xy, xz]
        
        
        # RPCA
        # run unrare data
        ctf_results = ctf(bt.copy(), 
                          mf.copy().reindex(bt.ids()),
                          'host_subject_id',
                          'timepoint',
                          feature_metadata=taxonomy)
        # rarefy data and rerun
        ctf_results_rare = ctf(rare_bt.copy(), 
                               mf.copy().reindex(rare_bt.ids()),
                               'host_subject_id',
                               'timepoint',
                               feature_metadata=taxonomy)
        # qc the data
        t_, p_, xy, _, xz, _, _, _ = qc_rarefaction(bt, ctf_results_rare[2], 
                                                    ctf_results[2], 
                                                    return_mantel=True,
                                                    mantel_permutations=1)
        # save the results
        test_results[(i, 'CTF')] = [t_, p_, xy, xz]
        
        
    columns_ = ['steiger_t_coeff', 'steiger_p',
                'rarefaction_mantel_coeff', 
                'no_rarefaction_mantel_coeff']
    test_results = pd.DataFrame(test_results, columns_).T
    test_results.index.name = 'iteration'
    return test_results

# import table(s)
ibd_table = load_table('IBD-2538/data/table.biom')
# import taxonomy
taxonomy = pd.read_csv('IBD-2538/data/taxonomy.tsv', sep='\t',
                       index_col=0, dtype={'sample_name':'str'})
# we don't filter in the tutorial but we need to here 
# since we are going to rarefy we want the same samples
# between the tables w/ and w/o rarefy.
# the rare depth was chosen from the origonal paper
# https://github.com/knightlab-analyses/longitudinal-ibd/blob/master/notebooks/01.1-setup.ipynb
# https://gut.bmj.com/content/67/9/1743
rare_depth = 7400
ibd_table = rpca_table_processing(ibd_table, 
                                  min_sample_count=rare_depth + 1,
                                  min_feature_count=2,  
                                  min_feature_frequency=0)
# import metadata
ibd_metadata = pd.read_csv('IBD-2538/data/metadata.tsv', sep='\t',
                       dtype={'sample_name':'str'}).set_index('sample_name')
ibd_metadata = ibd_metadata.reindex(ibd_table.ids())
# run example
example_results = rarefaction_compare_ibd_w_ctf(ibd_table.copy(), 
                                                ibd_metadata.copy(), 
                                                taxonomy.copy(),
                                                rare_depth)
example_results.reset_index().rename({'level_0':'fold', 'level_1':'method'}, axis=1).to_csv('rarefaction-IBD-2538-example.csv')

