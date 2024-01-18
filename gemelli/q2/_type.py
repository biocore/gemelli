from qiime2.plugin import SemanticType, model
from q2_types.sample_data import SampleData
from q2_types.feature_data import FeatureData


SampleTrajectory = SemanticType(
    'SampleTrajectory', variant_of=SampleData.field['type'])
FeatureTrajectory = SemanticType(
    'FeatureTrajectory', variant_of=FeatureData.field['type'])
CrossValidationResults = SemanticType(
    'CrossValidationResults', variant_of=SampleData.field['type'])
OrdinationCorrelation = SemanticType(
    'OrdinationCorrelation', variant_of=FeatureData.field['type'])
