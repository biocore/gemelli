import qiime2.plugin.model as model
from qiime2.plugin import ValidationError


class TrajectoryFormat(model.TextFileFormat):
    def _validate(self, n_records=None):
        with self.open() as fh:
            # check the header column names
            header = fh.readline()
            comp_columns = [i for i, head in enumerate(header.split('\t'))
                            if 'PC' in head]
            # ensure there at least two components
            if len(comp_columns) < 1:
                raise ValidationError('No PC# columns present. '
                                      'There should be at least one PC#'
                                      '(i.e. at minimum PC1) in trajectory.')
            # validate the body of the data
            for line_number, line in enumerate(fh, start=2):
                cells = line.split('\t')
                pc_type = [is_float(cells[c].strip()) for c in comp_columns]
                if not all(pc_type):
                    raise ValidationError('Non float values in trajectory.')
                if n_records is not None and (line_number - 1) >= n_records:
                    break

    def _validate_(self, level):
        record_count_map = {'min': 5, 'max': None}
        self._validate(record_count_map[level])


def is_float(str):
    try:
        float(str)
        return True
    except ValueError:
        return False


TrajectoryDirectoryFormat = model.SingleFileDirectoryFormat(
    'TrajectoryDirectoryFormat', 'trajectory.tsv',
    TrajectoryFormat)
