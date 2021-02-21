from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter

from autosklearn.pipeline.components.base import \
    AutoSklearnClassificationAlgorithm
from autosklearn.pipeline.constants import DENSE, UNSIGNED_DATA, PREDICTIONS


class LGBMClassifier(AutoSklearnClassificationAlgorithm):
    def __init__(self,
                 learning_rate=0.05,
                 num_leaves=31,
                 min_data_in_leaf=20,
                 feature_fraction=1.0,
                 n_estimators=100,
                 random_state=None):
        self.estimator = None
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.feature_fraction = feature_fraction
        self.min_data_in_leaf = min_data_in_leaf
        self.n_estimators = n_estimators

    def fit(self, X, Y):
        from lightgbm import LGBMClassifier

        self.estimator = LGBMClassifier(
            boosting_type='gbdt',
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            feature_fraction=self.feature_fraction,
            min_data_in_leaf=self.min_data_in_leaf,
            n_estimators=self.n_estimators,
        )

        self.estimator.fit(X, Y)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict(X)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()

        return self.estimator.predict_proba(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'LGBM',
                'name': 'Light Gradient Boosting Machine',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'handles_multioutput': False,
                'is_deterministic': True,
                'input': (DENSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        learning_rate = UniformFloatHyperparameter(
            "learning_rate", lower=5e-3, upper=0.2, default_value=0.05, log=True)
        num_leaves = UniformIntegerHyperparameter(
            "num_leaves", lower=16, upper=96, default_value=31)
        feature_fraction = UniformFloatHyperparameter(
            "feature_fraction", lower=0.75, upper=1.0, default_value=1.0)
        min_data_in_leaf = UniformIntegerHyperparameter(
            "min_data_in_leaf", lower=2, upper=30, default_value=20)
        n_estimators = UniformIntegerHyperparameter(
            "n_estimators", lower=50, upper=150, default_value=100)

        cs.add_hyperparameters([learning_rate, num_leaves, feature_fraction, min_data_in_leaf,
                                n_estimators])

        return cs
