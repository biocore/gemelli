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

    def transform(self):
        """ return loadings
        """
        return self.sample_loading, \
            self.feature_loading, \
            self.conditional_loading


class _BaseConstruct(object):

    """Base class for transformation/norm methods.
    Warning: This class should not be used directly.
    Use derived classes instead.
    """
    @abstractmethod
    def construct(self):
        """ Placeholder for construct this
        should be implemetned by sub-method"""
