import numpy as np

import pandas as pd

import pytest

import sklearn.datasets
import sklearn.model_selection

from autosklearn.data.validation import InputValidator


@pytest.mark.parametrize('openmlid', [2, 40975, 40984])
@pytest.mark.parametrize('as_frame', [True, False])
def test_data_validation_for_classification(openmlid, as_frame):
    x, y = sklearn.datasets.fetch_openml(data_id=openmlid, return_X_y=True, as_frame=as_frame)
    validator = InputValidator()

    if as_frame:
        # NaN is not supported in categories, so
        # drop columns with them.
        nan_cols = [i for i in x.columns if x[i].isnull().any()]
        cat_cols = [i for i in x.columns if x[i].dtype.name in ['category', 'bool']]
        unsupported_columns = list(set(nan_cols) & set(cat_cols))
        if len(unsupported_columns) > 0:
            x.drop(unsupported_columns, axis=1, inplace=True)

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        x, y, test_size=0.33, random_state=0)

    validator.fit(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                  is_classification=True)

    X_train_t, y_train_t = validator.transform(X_train, y_train)
    assert np.shape(X_train) == np.shape(X_train_t)

    # Leave columns that are complete NaN
    # The sklearn pipeline will handle that
    if as_frame and np.any(pd.isnull(X_train).values.all(axis=0)):
        assert np.any(pd.isnull(X_train_t).values.all(axis=0))
    elif not as_frame and np.any(pd.isnull(X_train).all(axis=0)):
        assert np.any(pd.isnull(X_train_t).all(axis=0))

    # make sure everything was encoded to number
    assert np.issubdtype(X_train_t.dtype, np.number)
    assert np.issubdtype(y_train_t.dtype, np.number)

    # Categorical columns are sorted to the beginning
    if as_frame:
        validator.feature_validator.feature_types is not None
        ordered_unique_elements = list(dict.fromkeys(validator.feature_validator.feature_types))
        if len(ordered_unique_elements) > 1:
            assert ordered_unique_elements[0] == 'categorical'


@pytest.mark.parametrize('openmlid', [505, 546, 531])
@pytest.mark.parametrize('as_frame', [True, False])
def test_data_validation_for_regression(openmlid, as_frame):
    x, y = sklearn.datasets.fetch_openml(data_id=openmlid, return_X_y=True, as_frame=as_frame)
    validator = InputValidator()

    if as_frame:
        # NaN is not supported in categories, so
        # drop columns with them.
        nan_cols = [i for i in x.columns if x[i].isnull().any()]
        cat_cols = [i for i in x.columns if x[i].dtype.name in ['category', 'bool']]
        unsupported_columns = list(set(nan_cols) & set(cat_cols))
        if len(unsupported_columns) > 0:
            x.drop(unsupported_columns, axis=1, inplace=True)

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        x, y, test_size=0.33, random_state=0)

    validator.fit(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                  is_classification=False)

    X_train_t, y_train_t = validator.transform(X_train, y_train)
    assert np.shape(X_train) == np.shape(X_train_t)

    # Leave columns that are complete NaN
    # The sklearn pipeline will handle that
    if as_frame and np.any(pd.isnull(X_train).values.all(axis=0)):
        assert np.any(pd.isnull(X_train_t).values.all(axis=0))
    elif not as_frame and np.any(pd.isnull(X_train).all(axis=0)):
        assert np.any(pd.isnull(X_train_t).all(axis=0))

    # make sure everything was encoded to number
    assert np.issubdtype(X_train_t.dtype, np.number)
    assert np.issubdtype(y_train_t.dtype, np.number)

    # Categorical columns are sorted to the beginning
    if as_frame:
        validator.feature_validator.feature_types is not None
        ordered_unique_elements = list(dict.fromkeys(validator.feature_validator.feature_types))
        if len(ordered_unique_elements) > 1:
            assert ordered_unique_elements[0] == 'categorical'
