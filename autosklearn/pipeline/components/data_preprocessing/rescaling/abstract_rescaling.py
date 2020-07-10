from ConfigSpace.configuration_space import ConfigurationSpace
from autosklearn.util.common import print_getrusage


class Rescaling(object):
    # Rescaling does not support fit_transform (as of 0.19.1)!

    def fit(self, X, y=None):
        print_getrusage(f"Preprocesort {self.preprocessor} start")

        self.preprocessor.fit(X)
        print_getrusage(f"Preprocesort {self.preprocessor} end")
        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        return cs
