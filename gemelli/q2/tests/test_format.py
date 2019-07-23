from skbio.util import get_data_path
from qiime2.plugin import ValidationError
from qiime2.plugin.testing import TestPluginBase
from gemelli.q2._format import TrajectoryFormat


class TestTrajectoryFormatTest(TestPluginBase):
    package = "q2_longitudinal.tests"

    def test_valid_simple(self):
        filepath = get_data_path('trajectory.tsv')
        format = TrajectoryFormat(filepath, mode='r')

        format.validate('min')
        format.validate('max')

    def test_valid_real_data(self):
        filepath = get_data_path('context-subject-ordination.tsv')
        format = TrajectoryFormat(filepath, mode='r')

        format.validate('min')
        format.validate('max')

    def test_invalid_header(self):
        filepath = get_data_path('trajectory-invalid-header.tsv')
        format = TrajectoryFormat(filepath, mode='r')
        for level in 'min', 'max':
            with self.assertRaises(ValidationError):
                format.validate(level)

    def test_invalid_non_numeric_column(self):
        filepath = get_data_path('trajectory-non-numeric.tsv')
        format = TrajectoryFormat(filepath, mode='r')
        for level in 'min', 'max':
            with self.assertRaises(ValidationError):
                format.validate(level)
