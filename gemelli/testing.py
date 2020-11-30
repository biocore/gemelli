# ----------------------------------------------------------------------------
# Copyright (c) 2019--, gemelli development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

import unittest
from pandas.testing import assert_series_equal


def assert_ordinationresults_equal(o1, o2, precision=5, verbose=False):
    """Asserts rough equality of OrdinationResults.

    Parameters
    ----------
        o1: skbio.OrdinationResults
        o2: skbio.OrdinationResults
            The values of these two objects will be compared.
            The following guidelines are followed when comparing o1.features
            and o2.features, and when comparing o1.samples and o2.samples:

                -The index names must be equal, but they can be in any order
                 (they'll be sorted before we compare the DataFrames). However,
                 it is NOT OK for these indices to contain repeated index
                 names; if this is the case, this function will fail.

                -The number of columns must be the same, but their names don't
                 matter.

                -For each column (aka pandas Series) within the DataFrames, we
                 run the pandas.testing.assert_series_equal() function (so
                 o1.features' first column is compared with o2.features' first
                 column, and so on). This function is called with
                 check_names=False (since we allow column names to be
                 different) and check_less_precise=[precision] (where
                 [precision] is the optional argument to this function
                 described below). We also wrap this invocation within a
                 try/except block, to allow for the following:

                -If the first invocation of assert_series_equal() fails between
                 two columns, we try calling it again with one of the columns
                 negated. This is because negating an entire axis in an
                 ordination doesn't really change the interpretation; see
                 @cameronmartino's comments on PR #29 in the biocore/DEICODE
                 repo for a better explanation of this.

        precision: int
            The argument to be passed to pandas.testing.assert_series_equal()
            when comparing the columns within o1 and o2's feature and sample
            DataFrames.

        verbose: bool
            If True, will print some information as this goes through the
            columns.

    """
    # This is just a tuple where the DataFrames within each 2-tuple within
    # it will be compared
    res_exp = ((o1.features, o2.features, "feature"),
               (o1.samples, o2.samples, "sample"))

    for (res, exp, aspect) in res_exp:
        # Just to protect ourselves, ensure that the indices only contain
        # unique names.
        assert len(set(res.index)) == len(res.index)
        assert len(set(exp.index)) == len(exp.index)
        # Row order doesn't matter, but the row names should match
        assert set(res.index) == set(exp.index)
        # Column names don't matter, but order does
        assert len(res.columns) == len(exp.columns)
        # Now, we can actually check that the ordination values match.
        # (First, we sort so that the rows in both the result and expected
        # DFs are in the same order, to enable us to use check_names=False
        # when calling assert_series_equal().)
        res_sorted = res.sort_index()
        exp_sorted = exp.sort_index()
        for col_index in range(len(res_sorted.columns)):
            # Extract the n-th column (a pandas Series) from both the
            # result and expected DataFrame, then compare their values.
            res_series = res_sorted.iloc[:, col_index]
            exp_series = exp_sorted.iloc[:, col_index]
            try:
                # First, just try comparing the two PCs and seeing if their
                # values are approximately equal.
                assert_series_equal(res_series, exp_series, check_names=False,
                                    check_less_precise=precision)
            except AssertionError:
                # It's fine for any of the "PC"s (i.e. columns in the
                # OrdinationResults) to be off by a factor of -1, since
                # that doesn't really change the interpretation of anything
                # (c/o @cameronmartino's comment in #29).
                # To allow for this case to pass the tests, we just try
                # negating one of the series, and seeing if
                # that makes them approximately equal.
                # (If they're *still* not equal, this test will fail.)
                assert_series_equal(-res_series, exp_series, check_names=False,
                                    check_less_precise=precision)
            if verbose:
                print("PC {} for the {} ordination matches.".format(col_index,
                                                                    aspect))


class CliTestCase(unittest.TestCase):

    def assertExitCode(self, value, result):
        try:
            self.assertEqual(0, result.exit_code)
        except AssertionError:
            ex = result.exception
            error = Exception('Command failed with non-zero exit code')
            raise error.with_traceback(ex.__traceback__)


def absolute_sort(matt):
    matt = abs(matt)
    matt = matt[matt[:, 0].argsort()]
    return matt
