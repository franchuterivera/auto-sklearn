import functools
import typing
import warnings

import numpy as np

import pandas as pd
from pandas.api.types import is_numeric_dtype

import scipy.sparse

import sklearn.utils
from sklearn import preprocessing
from sklearn.base import BaseEstimator
from sklearn.compose import make_column_transformer


class FeatureValidator(BaseEstimator):

    def __init__(self) -> None:
        # If a dataframe was provided, we populate
        # this attribute with the column types from the dataframe
        # That is, this attribute contains whether autosklearn
        # should treat a column as categorical or numerical
        # During fit, if the user provided feat_types, the user
        # constrain is honored. If not, this attribute is used.
        self.feature_types = None  # type: typing.Optional[typing.List[str]]

        self.data_type = None  # type: typing.Optional[type]

        self.encoder = None  # type: typing.Optional[BaseEstimator]
        self.enc_columns = []  # type: typing.List[str]

        self._is_fitted = False

    def fit(
        self,
        X_train: typing.Union[np.ndarray, pd.DataFrame],
        X_test: typing.Optional[typing.Union[np.ndarray, pd.DataFrame]] = None,
        feat_type: typing.Optional[typing.List[str]] = None,
    ) -> BaseEstimator:
        """
        Validates and fit a categorical encoder (if needed) to the features.
        The supported data types are List, numpy arrays and pandas DataFrames.
        CSR sparse data types are also supported

        Parameters
        ----------
            X_train: typing.Union[np.ndarray, pd.DataFrame]
                A set of features that are going to be validated (type and dimensionality
                checks) and a encoder fitted in the case the data needs encoding
            X_test: typing.Union[np.ndarray, pd.DataFrame]
                A hold out set of data used for checking
            feat_type: typing.Optional[typing.List[str]]
                In case the data is not a pandas DataFrame, this list indicates
                which columns should be treated as categorical
        """
        # Check that the data is valid
        self.check_data(X_train, X_test, feat_type)

        # Fit on the training data
        self._fit(X_train)

        self._is_fitted = True

        return self

    def _fit(
        self,
        X: typing.Union[np.ndarray, pd.DataFrame],
    ) -> BaseEstimator:
        """
        In case input data is a pandas DataFrame, this utility encodes the user provided
        features (from categorical for example) to a numerical value that further stages
        will be able to use

        Parameters
        ----------
            X: typing.Union[np.ndarray, pd.DataFrame]
                A set of features that are going to be validated (type and dimensionality
                checks) and a encoder fitted in the case the data needs encoding
        """
        if hasattr(X, "iloc") and not scipy.sparse.issparse(X):
            # Treat a column with all instances a NaN as numerical
            if np.any(pd.isnull(X)):
                for column in X.columns:
                    if X[column].isna().all():
                        X[column] = pd.to_numeric(X[column])

            self.enc_columns, self.feature_types = self._get_columns_to_encode(X)

            if len(self.enc_columns) > 0:

                self.encoder = make_column_transformer(
                    (preprocessing.OrdinalEncoder(), self.enc_columns),
                    remainder="passthrough"
                )

                # Mypy redefinition
                assert self.encoder is not None
                self.encoder.fit(X)

                # The column transformer reoders the feature types - we therefore need to change
                # it as well
                def comparator(cmp1: str, cmp2: str) -> int:
                    if (
                        cmp1 == 'categorical' and cmp2 == 'categorical'
                        or cmp1 == 'numerical' and cmp2 == 'numerical'
                    ):
                        return 0
                    elif cmp1 == 'categorical' and cmp2 == 'numerical':
                        return -1
                    elif cmp1 == 'numerical' and cmp2 == 'categorical':
                        return 1
                    else:
                        raise ValueError((cmp1, cmp2))
                self.feature_types = sorted(
                    self.feature_types,
                    key=functools.cmp_to_key(comparator)
                )
        return self

    def transform(
        self,
        X: typing.Union[np.ndarray, pd.DataFrame],
    ) -> np.ndarray:
        """
        Validates and fit a categorical encoder (if needed) to the features.
        The supported data types are List, numpy arrays and pandas DataFrames.

        Parameters
        ----------
            X_train: typing.Union[np.ndarray, pd.DataFrame]
                A set of features, whose categorical features are going to be
                transformed

        Return
        ------
            np.ndarray:
                The transformed array
        """
        if not self._is_fitted:
            raise ValueError("Cannot call transform on a validator that is not fitted")

        # Pandas related transformations
        if hasattr(X, "iloc") and self.encoder is not None:
            if np.any(pd.isnull(X)):
                # After above check it means that if there is a NaN
                # the whole column must be NaN
                # Make sure it is numerical and let the pipeline handle it
                for column in X.columns:
                    if X[column].isna().all():
                        X[column] = pd.to_numeric(X[column])
            try:
                X = self.encoder.transform(X)
            except ValueError as e:
                if 'Found unknown categories' in e.args[0]:
                    # Make the message more informative
                    raise ValueError(
                        "During fit, the input features contained categorical values in columns"
                        "{}, with categories {} which were encoded by Auto-sklearn automatically."
                        "Nevertheless, a new input contained new categories not seen during "
                        "training = {}. The OrdinalEncoder used by Auto-sklearn cannot handle "
                        "this yet (due to a limitation on scikit-learn being addressed via:"
                        " https://github.com/scikit-learn/scikit-learn/issues/17123)"
                        "".format(
                            self.enc_columns,
                            self.encoder.transformers_[0][1].categories_,
                            e.args[0],
                        )
                    )
                else:
                    raise e

        # Sparse related transformations
        if scipy.sparse.issparse(X):
            X.sort_indices()

        return sklearn.utils.check_array(
            X,
            force_all_finite=False,
            accept_sparse='csr'
        )

    def check_data(
        self,
        X_train: typing.Union[np.ndarray, pd.DataFrame],
        X_test: typing.Optional[typing.Union[np.ndarray, pd.DataFrame]],
        feat_type: typing.Optional[typing.List[str]] = None,
    ) -> None:
        """
        Makes sure the features comply with auto-sklearn data requirements and
        checks if X_train/X_test dimensionality

        This method also stores metadata for future checks

        Parameters
        ----------
            X_train: typing.Union[np.ndarray, pd.DataFrame]
                A set of features that are going to be validated (type and dimensionality
                checks) and a encoder fitted in the case the data needs encoding
            X_test: typing.Union[np.ndarray, pd.DataFrame]
                A hold out set of data used for checking
            feat_type: typing.Optional[typing.List[str]]
                In case the data is not a pandas DataFrame, this list indicates
                which columns should be treated as categorical
        """
        # Register the user provided feature types
        if feat_type is not None:
            if hasattr(X_train, "iloc"):
                raise ValueError("When providing a DataFrame to Auto-Sklearn, we extract "
                                 "the feature types from the DataFrame.dtypes. That is, "
                                 "providing the option feat_type to the fit method is not "
                                 "supported when using a Dataframe. Please make sure that the "
                                 "type of each column in your DataFrame is properly set. "
                                 "More details about having the correct data type in your "
                                 "DataFrame can be seen in "
                                 "https://pandas.pydata.org/pandas-docs/stable/reference"
                                 "/api/pandas.DataFrame.astype.html")
            # Some checks if feat_type is provided
            if len(feat_type) != np.shape(X_train)[1]:
                raise ValueError('Array feat_type does not have same number of '
                                 'variables as X has features. %d vs %d.' %
                                 (len(feat_type), np.shape(X_train)[1]))
            if not all([isinstance(f, str) for f in feat_type]):
                raise ValueError('Array feat_type must only contain strings.')

            for ft in feat_type:
                if ft.lower() not in ['categorical', 'numerical']:
                    raise ValueError('Only `Categorical` and `Numerical` are '
                                     'valid feature types, you passed `%s`' % ft)

            # Here we register proactively the feature types for
            # Processing Numpy arrays
            self.feature_types = feat_type

        self._check_data(X_train)

        if X_test is not None:
            self._check_data(X_test)

            if np.shape(X_train)[1] != np.shape(X_test)[1]:
                raise ValueError("The feature dimensionality of the train and test "
                                 "data does not match train({}) != test({})".format(
                                     np.shape(X_train)[1],
                                     np.shape(X_test)[1]
                                 ))

    def _check_data(
        self,
        X: typing.Union[np.ndarray, pd.DataFrame],
    ) -> None:
        """
        Feature dimensionality and data type checks

        Parameters
        ----------
            X: typing.Union[np.ndarray, pd.DataFrame]
                A set of features that are going to be validated (type and dimensionality
                checks) and a encoder fitted in the case the data needs encoding
        """

        if not isinstance(X, (np.ndarray, pd.DataFrame, list)) and not scipy.sparse.issparse(X):
            raise ValueError("Auto-sklearn only supports Numpy arrays, Pandas DataFrames,"
                             " scipy sparse and Python Lists, yet, the provided input is"
                             " of type {}".format(
                                 type(X)
                             ))

        if isinstance(X, list):
            try:
                np.array(X).astype(np.float64)
            except ValueError as e:
                raise ValueError("When providing a list of features to Auto-sklearn, "
                                 "this list of elements must contain only numerical values "
                                 "as it will be converted to Numpy. Casting to Numpy failed "
                                 "with exception: {}".format(e))

        if self.data_type is None:
            self.data_type = type(X)
        if self.data_type != type(X):
            warnings.warn("Auto-sklearn previously received features of type %s "
                          "yet the current features have type %s. Changing the dtype "
                          "of inputs to an estimator might cause problems" % (
                                str(self.data_type),
                                str(type(X)),
                             ),
                          )

        # Do not support category/string numpy data. Only numbers
        if hasattr(X, "dtype"):
            if not np.issubdtype(X.dtype.type, np.number):  # type: ignore[union-attr]
                raise ValueError(
                    "When providing a numpy array to Auto-sklearn, the only valid "
                    "dtypes are numerical ones. The provided data type {} is not supported."
                    "".format(
                        X.dtype.type,  # type: ignore[union-attr]
                    )
                )

        # Then for Pandas, we do not support Nan in categorical columns
        if hasattr(X, "iloc"):
            enc_columns, _ = self._get_columns_to_encode(X)
            if len(enc_columns) > 0:
                if np.any(pd.isnull(
                    X[enc_columns].dropna(  # type: ignore[call-overload]
                        axis='columns', how='all')
                )):
                    # Ignore all NaN columns, and if still a NaN
                    # Error out
                    raise ValueError("Categorical features in a dataframe cannot contain "
                                     "missing/NaN values. The OrdinalEncoder used by "
                                     "Auto-sklearn cannot handle this yet (due to a "
                                     "limitation on scikit-learn being addressed via: "
                                     "https://github.com/scikit-learn/scikit-learn/issues/17123)"
                                     )
            if self.enc_columns and set(self.enc_columns) != set(enc_columns):
                raise ValueError(
                    "Changing the column-types of the input data to Auto-Sklearn is not "
                    "allowed. The estimator previously was fitted with categorical/boolean "
                    "columns {}, yet, the new input data has categorical/boolean values {}. "
                    "Please recreate the estimator from scratch when changing the input "
                    "data. ".format(
                        self.enc_columns,
                        enc_columns,
                    )
                )

    def _get_columns_to_encode(
        self,
        X: typing.Union[np.ndarray, pd.DataFrame],
    ) -> typing.Tuple[typing.List[str], typing.List[str]]:
        """
        Return the columns to be encoded from a pandas dataframe

        Parameters
        ----------
            X: typing.Union[np.ndarray, pd.DataFrame]
                A set of features that are going to be validated (type and dimensionality
                checks) and a encoder fitted in the case the data needs encoding
        Returns
        -------
            enc_columns:
                Columns to encode, if any
            Feature_types:
                Type of each column numerical/categorical
        """
        # Register if a column needs encoding
        enc_columns = []

        # Also, register the feature types for the estimator
        feature_types = []

        # Make sure each column is a valid type
        for i, column in enumerate(X.columns):
            if X[column].dtype.name in ['category', 'bool']:

                enc_columns.append(column)
                feature_types.append('categorical')
            # Move away from np.issubdtype as it causes
            # TypeError: data type not understood in certain pandas types
            elif not is_numeric_dtype(X[column]):
                if X[column].dtype.name == 'object':
                    raise ValueError(
                        "Input Column {} has invalid type object. "
                        "Cast it to a valid dtype before using it in Auto-Sklearn. "
                        "Valid types are numerical, categorical or boolean. "
                        "You can cast it to a valid dtype using "
                        "pandas.Series.astype ."
                        "If working with string objects, the following "
                        "tutorial illustrates how to work with text data: "
                        "https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html".format(  # noqa: E501
                            column,
                        )
                    )
                elif pd.core.dtypes.common.is_datetime_or_timedelta_dtype(
                    X[column].dtype
                ):
                    raise ValueError(
                        "Auto-sklearn does not support time and/or date datatype as given "
                        "in column {}. Please convert the time information to a numerical value "
                        "first. One example on how to do this can be found on "
                        "https://stats.stackexchange.com/questions/311494/".format(
                            column,
                        )
                    )
                else:
                    raise ValueError(
                        "Input Column {} has unsupported dtype {}. "
                        "Supported column types are categorical/bool/numerical dtypes. "
                        "Make sure your data is formatted in a correct way, "
                        "before feeding it to Auto-Sklearn.".format(
                            column,
                            X[column].dtype.name,
                        )
                    )
            else:
                feature_types.append('numerical')
        return enc_columns, feature_types
