from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter

from autosklearn.pipeline.components.base import \
    AutoSklearnClassificationAlgorithm
from autosklearn.pipeline.constants import DENSE, UNSIGNED_DATA, PREDICTIONS


class XGBClassifier(AutoSklearnClassificationAlgorithm):
    def __init__(self,
                 learning_rate=0.05,
                 n_estimators=100,
                 max_depth=3,
                 min_child_weight=1,
                 gamma=0.01,
                 subsample=1.0,
                 colsample_bytree=1.0,
                 reg_alpha=0.0,
                 reg_lambda=1.0,
                 random_state=None):
        self.estimator = None
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda

    def fit(self, X, Y):
        from xgboost import XGBClassifier

        self.estimator = XGBClassifier(
            n_estimators=self.n_estimators,
            booster='gbtree',
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_child_weight=self.min_child_weight,
            gamma=self.gamma,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
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
        return {'shortname': 'XGBClassifier',
                'name': 'XGBoost Classifier',
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
            "learning_rate", lower=5e-3, upper=0.2, default_value=0.1, log=True)
        n_estimators = UniformIntegerHyperparameter(
            "n_estimators", lower=512, upper=2000, default_value=1000)
        max_depth = UniformIntegerHyperparameter(
            "max_depth", lower=3, upper=20, default_value=3)
        min_child_weight = UniformIntegerHyperparameter(
           "min_child_weight", lower=1, upper=5, default_value=1)
        gamma = UniformFloatHyperparameter(
           "gamma", lower=0.0, upper=5.0, default_value=0.01)
        subsample = UniformFloatHyperparameter(
           "subsample", lower=0.5, upper=1.0, default_value=1.0)
        colsample_bytree = UniformFloatHyperparameter(
           "colsample_bytree", lower=0.5, upper=1.0, default_value=1.0)
        reg_alpha = UniformFloatHyperparameter(
           "reg_alpha", lower=0.0, upper=10.0, default_value=0.0)
        reg_lambda = UniformFloatHyperparameter(
           "reg_lambda", lower=0.0, upper=10.0, default_value=1.0)

        cs.add_hyperparameters([learning_rate, n_estimators, max_depth, min_child_weight,
                                gamma, subsample, colsample_bytree, reg_alpha, reg_lambda])

        return cs
