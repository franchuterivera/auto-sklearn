from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter

from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import DENSE, SPARSE, UNSIGNED_DATA, INPUT

from autosklearn.util.common import print_getrusage


class NumericalImputation(AutoSklearnPreprocessingAlgorithm):

    def __init__(self, strategy='mean', random_state=None):
        self.strategy = strategy
        self.random_state = random_state

    def fit(self, X, y=None):
        import sklearn.impute

        self.preprocessor = sklearn.impute.SimpleImputer(
            strategy=self.strategy, copy=False)
        print_getrusage("In Numerical inputation with {self.preprocessor} before fit")
        self.preprocessor.fit(X)
        print_getrusage("In Numerical inputation with {self.preprocessor} after fit")
        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        print_getrusage(f"In Numerical inputation with {self.preprocessor} before transform")
        new = self.preprocessor.transform(X)
        print_getrusage(f"In Numerical inputation  with {self.preprocessor} after transform")
        return new

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'NumericalImputation',
                'name': 'Numerical Imputation',
                'handles_missing_values': True,
                'handles_nominal_values': True,
                'handles_numerical_features': True,
                'prefers_data_scaled': False,
                'prefers_data_normalized': False,
                'handles_regression': True,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                # TODO find out if this is right!
                'handles_sparse': True,
                'handles_dense': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (INPUT,),
                'preferred_dtype': None}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        # TODO add replace by zero!
        strategy = CategoricalHyperparameter(
            "strategy", ["mean", "median", "most_frequent"], default_value="mean")
        cs = ConfigurationSpace()
        cs.add_hyperparameter(strategy)
        return cs
