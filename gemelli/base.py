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
        """ TODO
        """
        return self.sample_loading, self.feature_loading, self.time_loading


class _BaseTransform(object):

    """Base class for transformation/norm methods.
    Warning: This class should not be used directly.
    Use derived classes instead.
    """
    @abstractmethod
    def fit(self):
        """ Placeholder for fit this
        should be implemetned by sub-method"""

    def transform(self):
        """ TODO
        """
        return self.X_sp
