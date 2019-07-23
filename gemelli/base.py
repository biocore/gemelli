# ----------------------------------------------------------------------------
# Copyright (c) 2019--, gemelli development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from abc import abstractmethod


class _BaseImpute(object):

    """Base class for imputation methods.
    Warning: This class should not be used directly.
    Use derived classes instead.
    """
    @abstractmethod
    def fit(self):
        """ Placeholder for fit this
        should be implemetned by sub-method"""
    @abstractmethod
    def label(self):
        """ Placeholder for fit this
        should be implemetned by sub-method"""


class _BaseConstruct(object):

    """Base class for transformation/norm methods.
    Warning: This class should not be used directly.
    Use derived classes instead.
    """
    @abstractmethod
    def construct(self):
        """
        conditional_loading  : array-like or list of array-like
             The conditional loading vectors
             of shape (conditions, r) if there is 1 type
             of condition, and a list of such matrices if
             there are more than 1 type of condition
         feature_loading : array-like
             The feature loading vectors
             of shape (features, r)
         sample_loading : array-like
             The sample loading vectors
             of shape (samples, r) """
