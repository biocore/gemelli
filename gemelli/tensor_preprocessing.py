# ----------------------------------------------------------------------------
# Copyright (c) 2019--, gemelli development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

import numpy as np
import pandas as pd
from .utils import match
from .base import _BaseTransform
from deicode.preprocessing import rclr

@experimental(as_of="0.0.0")
class Build(_BaseTransform):

    def __init__(self, pseudocount=1):

        """

        Parameters
        ----------
        pseudocount : float, optional
            A pseudocount used in the
            case that a sample, feature,
             and/or condition is completely
             missing from a vector slice in 
             the tensor formed.

        """

        self.pseudocount = pseudocount

    @property
    def pseudocount(self):

        """

        pseudocount property
        allows pseudocount to
        be set explictly.

        """

        return self.pseudocount

    @pseudocount.setter
    def pseudocount(self, value):

        """

        Set pseudocount value
        for property.

        Parameters
        ----------
        value : float
            pseudocount value

        Raises
        ------
        ValueError
            Raises an error if value is less than zero.

        Examples
        --------
        >>> c = Celsius()
        >>> c.temperature
        >>> c.temperature = 37

        """

        if value < 0.0:
            raise ValueError("pseudocount should be larger than one")
        self.pseudocount = value

    def fit(self,table,mapping,ID_col,cond_col):

        """

        Description.

        Parameters
        ----------

        Returns
        -------

        Raises
        ------
        ValueError

        Warning

        References
        ----------
 
        Examples
        --------

        """


        self.table = table.copy()
        self.mapping = mapping.copy()
        self.ID_col = ID_col
        self.cond_col = cond_col
        self._fit()

        return self

    def fit_transform(self,table,mapping,ID_col,cond_col):

        """

        Description.

        Parameters
        ----------

        Returns
        -------

        Raises
        ------
        ValueError

        Warning

        References
        ----------
 
        Examples
        --------

        """


        self.table = table.copy()
        self.mapping = mapping.copy()
        self.ID_col = ID_col
        self.cond_col = cond_col
        self._fit()

        return self.T , self.TRCLR, self.ID_order, self.feature_order, self.cond_order

    def _fit(self):

        """

        Description.

        Parameters
        ----------

        Returns
        -------

        Raises
        ------
        ValueError

        Warning

        References
        ----------
 
        Examples
        --------

        """

        # check that all indicies match & are unqiue 
        self.table,self.mapping = match(self.table,self.mapping)

        # order ids, cond, feats
        ID_order = sorted(set(self.mapping[self.ID_col]))
        cond_order = sorted(set(self.mapping[self.cond_col]))

        # empty Tensor to fill 
        Tensor = np.zeros((len(cond_order),
                           len(self.table.columns),
                           len(ID_order)))

        # fill tensor where possible
        table_index = np.array(self.table.index)
        table_array = self.table.values
        num_missing = 0 # check if fully missing samples
        for i,c_i in enumerate(cond_order):
            for j,ID_j in enumerate(ID_order):
                # get index ID assoc. in cond.
                idx = set(self.mapping[(self.mapping[self.ID_col].isin([ID_j])) \
                                & (self.mapping[self.cond_col].isin([c_i]))].index)
                if len(idx)>1:
                    warnings.warn('',join(["Condition ",str(c_i),
                                        " has multiple sample ",
                                        "with the same IDs ",
                                        str(ID_j)]), RuntimeWarning)
                elif len(idx)==0:
                    num_missing+=1
                    continue 
                # fill slice 
                Tensor[i,:,j] = table_array[table_index == list(idx),:].sum(axis=0)
                
        # find percent totally missing samples
        self.perc_missing = num_missing/(len(cond_order)*len(ID_order))
        if self.perc_missing > 0.50:
            warnings.warn(''.join(["Total Missing Sample Exceeds 50% ",
                        "some conditions or samples may ",
                        "need to be removed."]), RuntimeWarning)

        # perform RCLR transformation
        self.Tensor = Tensor
        self.tensor_rclr()

        # save intermediates
        self.ID_order  = ID_order
        self.feature_order  = self.table.columns
        self.cond_order  = cond_order

    def tensor_rclr(self):

        """

        Tensor wrapped for rclr transform
        will add pseudocount where samples
        are completely missing. The transform
        of those samples will be zero again.

        Parameters
        ----------

        self.Tensor array-like
            (condition, features, samples)

        Returns
        -------

        Raises
        ------
        ValueError

        Warning

        References
        ----------
 
        Examples
        --------

        """

        if len(self.Tensor.shape) != 3:
            raise ValueError('Array Contains Negative Values')

        if (self.Tensor < 0).any():
            raise ValueError('Array Contains Negative Values')

        if np.count_nonzero(np.isinf(self.Tensor)) != 0:
            raise ValueError('Data-table contains either np.inf or -np.inf')

        if np.count_nonzero(np.isnan(self.Tensor)) != 0:
            raise ValueError('Data-table contains nans')

        if np.count_nonzero(self.Tensor) == 0:
            warnings.warn("Data-table contains no zeros.", RuntimeWarning)
        
        # copy tensor to transform
        TRCLR = self.Tensor.copy()

        # pseudocount totally missing samp: 
        # sum of all feat (time,samp)==0
        if self.perc_missing > 0.0:
            for i,j in np.argwhere(self.Tensor.sum(axis=1) == 0):
                self.Tensor[i,:,j]+=self.pseudocount
        # add for any totally zero features (should not occur)
        if sum(self.Tensor.sum(axis=0).sum(axis=1)==0) > 0:
            self.Tensor[:,self.Tensor.sum(axis=0).sum(axis=1)==0,:] += self.pseudocount
        # add for any totally zero timepoint (should not occur)
        if sum(self.Tensor.sum(axis=2).sum(axis=1)==0) > 0:
            self.Tensor[self.Tensor.sum(axis=2).sum(axis=1)==0,:,:] += self.pseudocount

        # flatten
        TRCLR = np.concatenate([self.Tensor[i,:,:].T 
                                for i in range(self.Tensor.shape[0])],axis=0)

        # transform flat
        TRCLR = rclr().fit_transform(TRCLR)

        # re-shape tensor
        TRCLR = np.dstack([TRCLR[(i-1)*self.Tensor.shape[-1]\
                                  :(i)*self.Tensor.shape[-1]] 
                           for i in range(1,self.Tensor.shape[0]+1)])

        # fill nan with zero
        TRCLR[np.isnan(TRCLR)] = 0 

        self.TRCLR = TRCLR
