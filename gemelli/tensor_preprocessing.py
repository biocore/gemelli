import numpy as np
import pandas as pd
from .utils import match
from deicode.preprocessing import rclr


class table_to_tensor(_BaseTransform):

    def __init__(self):
        """
        TODO
        """

    def fit(self,table,mapping,IDcol,timecol,
            filter_timepoints=True,filter_samples=False,
            min_sample_count = 0,min_feature_count = 0):
        """  TODO """
        self.table = table.copy()
        self.mapping = mapping.copy()
        self.IDcol = IDcol
        self.timecol = timecol
        self.filter_timepoints = filter_timepoints
        self.filter_samples = filter_samples
        self.min_sample_count = min_sample_count
        self.min_feature_count = min_feature_count
        self._fit()
        return self

    def fit_transform(self,table,mapping,IDcol,timecol,
                      filter_timepoints=True,filter_samples=False,
                      min_sample_count = 0,min_feature_count = 0):
        """ TODO  """
        self.table = table.copy()
        self.mapping = mapping.copy()
        self.IDcol = IDcol
        self.timecol = timecol
        self.filter_timepoints = filter_timepoints
        self.filter_samples = filter_samples
        self.min_sample_count = min_sample_count
        self.min_feature_count = min_feature_count
        self._fit()
        return self.T, (self.tensor_columns self.tensor_index, elf.tensor_time), self.mapping_time
        
    def _fit(self):
        """
        TODO
        """
        # cannot have both
        if sum([self.filter_timepoints,self.filter_samples])>1:
            raise ValueError('Must choose to replace samples by (t-1),'
                            ' filter by missing samples or timepoints'
                            ' not multiple.')
        # filter cutoffs
        self.table = self.table.T[self.table.sum() > self.min_feature_count]
        self.table = self.table.T[self.table.sum() > self.min_sample_count]

        # filter setup 
        # table by timepoint
        mapping_time = {k:df for k,df in self.mapping.groupby(self.timecol)}
        # remove timepoint with missing samples
        drop = {k_:[v_ 
                for v_ in list(set(self.mapping[self.IDcol])-set(df_[self.IDcol]))] 
                for k_,df_ in mapping_time.items()}

        # sample-removal
        if self.filter_samples==True:
            self.mapping = self.mapping[~self.mapping[self.IDcol].isin([v_ 
                                                    for k,v in drop.items() 
                                                    for v_ in v])]
        # timepoint-removal
        elif self.filter_timepoints==True:
            self.mapping = self.mapping[~self.mapping[self.timecol].isin([k for k,v in drop.items() 
                                                    if len(v)!=0])]
        else:
            raise ValueError('Must choose to replace samples by (t-1),'
                            ' filter by missing samples or timepoints.'
                            ' All of them can _not_ be False')  

        # remove zero sum features across flattened tensor b4 rclr
        T, mapping_time,table_tmp = reshape_tensor(self.table.copy(),self.mapping,self.timecol,self.IDcol)
        T_filter = np.concatenate([T[i,:,:].T for i in range(T.shape[0])],axis=0)
        sum_zero = [table_tmp.columns[i] for i, x in enumerate(list(T_filter.sum(axis=0))) if x == 0]
        self.table = self.table.drop(sum_zero,axis=1)
        T, mapping_time,table_tmp = reshape_tensor(self.table.copy(),self.mapping,self.timecol,self.IDcol)

        #test for zeros
        if any(~(np.concatenate([T[i,:,:].T for i in range(T.shape[0])],axis=0).sum(axis=1) > 0)):
            raise ValueError('Some samples sum to zero,'
                            ' consider increasing the sample'
                            ' read count cutoff')
        elif any(~(np.concatenate([T[i,:,:].T for i in range(T.shape[0])],axis=0).sum(axis=0) > 0)):
            raise ValueError('Some features sum to zero,'
                            ' consider increasing the feature'
                            ' read count cutoff')
        # if passed zero check
        self.tensor = T
        self.mapping_time = mapping_time
        self.tensor_columns = table_tmp.columns
        self.tensor_index = table_tmp.index
        self.tensor_time = sorted(list(mapping_time.keys()))

def reshape_tensor(table,mapping,timecol,IDcol):
    """ 
    Restructure dataframe into tensor 
    by metadata IDs and timepoints
    """
    # table by timepoint
    mapping_time = {k:df for k,df in mapping.groupby(timecol)}
    # create tensor
    tensor_stack = []
    # wort in numerical order
    for timepoint in sorted(mapping_time.keys()):
        # get table timepoint
        table_tmp,meta_tmp = match(table,mapping_time[timepoint])
        # fix ID cols
        meta_tmp.set_index(IDcol,inplace=True,drop=True)
        # sort so all ID match
        table_tmp = table_tmp.T.sort_index().T
        # check to make sure id's are unique to each time
        if len(meta_tmp.index.get_duplicates()):
            idrep = [str(idrep) for idrep in meta_tmp.index.get_duplicates()]
            idrep = ', '.join(idrep)
            raise ValueError('At timepoint '+str(timepoint)+
                             'The ids '+idrep+' are repeated.'
                            ' Please provide unique IDs to each time.')
        # index match
        table_tmp.index = meta_tmp.index
        table_tmp.sort_index(inplace=True)
        meta_tmp.sort_index(inplace=True)
        # update mapping time
        mapping_time[timepoint] = meta_tmp
        tensor_stack.append(table_tmp)
    # return both tensor and time_metadata dict, table for ref.
    return np.dstack(tensor_stack).T,mapping_time,table_tmp

def tensor_rclr(T):
    """ 
    Tensor wrapped for deicode rclr transform
    """
    # flatten, transform, and reshape 
    T_rclr = np.concatenate([T[i,:,:].T 
                             for i in range(T.shape[0])],axis=0)
    T_rclr = rclr().fit_transform(T_rclr)
    T_rclr = np.dstack([T_rclr[(i-1)*T.shape[-1]:(i)*T.shape[-1]] 
                        for i in range(1,T.shape[0]+1)])
    T_rclr[np.isnan(T_rclr)] = 0 
    return T_rclr