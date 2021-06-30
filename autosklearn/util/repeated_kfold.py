#  type: ignore
#  Mypy Bug: error: Cannot determine type of 'n_repeats'  [has-type]
import copy
import numbers
from abc import ABCMeta
from typing import Any, Generator, List, Optional

import numpy as np

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils import check_random_state


class _RepeatedMultiSplits(metaclass=ABCMeta):
    """Repeated splits for an arbitrary randomized CV splitter.
    Repeats splits for cross-validators n times with different randomization
    in each repetition.
    Parameters
    ----------
    cv : callable
        Cross-validator class.
    n_repeats : int, default=10
        Number of times cross-validator needs to be repeated.
    random_state : int, RandomState instance or None, default=None
        Passes `random_state` to the arbitrary repeating cross validator.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.
    **cvargs : additional params
        Constructor parameters for cv. Must not contain random_state
        and shuffle.
    """
    def __init__(self, cv: StratifiedKFold, *,
                 n_repeats: int = 10, n_splits: List[int] = [3, 5, 10],
                 random_state: Optional[np.random.RandomState] = None, **cvargs: Any) -> None:
        if not isinstance(n_repeats, numbers.Integral):
            raise ValueError("Number of repetitions must be of Integral type.")

        if n_repeats <= 0:
            raise ValueError("Number of repetitions must be greater than 0.")

        if any(key in cvargs for key in ('random_state', 'shuffle')):
            raise ValueError(
                "cvargs must not contain random_state or shuffle.")

        self.cv = cv
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.cvargs = cvargs

    def split(self, X: np.ndarray, y: Optional[np.ndarray] = None,
              groups: Optional[np.ndarray] = None) -> Generator:
        """Generates indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            The target variable for supervised learning problems.
        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        rng = check_random_state(self.random_state)

        for this_split in self.n_splits:
            this_cvargs = copy.deepcopy(self.cvargs)
            this_cvargs['n_splits'] = this_split
            cv = self.cv(random_state=rng, shuffle=True,
                         **this_cvargs)
            for train_index, test_index in cv.split(X, y, groups):
                yield train_index, test_index

    def get_n_splits(self, X: Optional[np.ndarray] = None,
                     y: Optional[np.ndarray] = None,
                     groups: List[np.ndarray] = None) -> int:
        """Returns the number of splitting iterations in the cross-validator
        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.
            ``np.zeros(n_samples)`` may be used as a placeholder.
        y : object
            Always ignored, exists for compatibility.
            ``np.zeros(n_samples)`` may be used as a placeholder.
        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        cv_n_splits = []
        rng = check_random_state(self.random_state)
        for this_split in self.n_splits:
            this_cvargs = copy.deepcopy(self.cvargs)
            this_cvargs['n_splits'] = this_split
            cv = self.cv(random_state=rng, shuffle=True,
                         **this_cvargs)
            cv_n_splits.append(cv.get_n_splits(X, y, groups))
        return sum(cv_n_splits)


class RepeatedStratifiedMultiKFold(_RepeatedMultiSplits):
    """Repeated Stratified Multi-K-Fold cross validator.
    Repeats Stratified K-Fold n times with different randomization in each
    repetition, for multiple k-splits.
    Read more in the :ref:`User Guide <repeated_k_fold>`.
    Parameters
    ----------
    n_splits : List[int], default=[3, 5, 10]
        Number of folds. Must be at least 2.
    n_repeats : int, default=1
        Number of times cross-validator needs to be repeated.
    random_state : int, RandomState instance or None, default=None
        Controls the generation of the random states for each repetition.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import RepeatedStratifiedKFold
    >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    >>> y = np.array([0, 0, 1, 1])
    >>> rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=2,
    ...     random_state=36851234)
    >>> for train_index, test_index in rskf.split(X, y):
    ...     print("TRAIN:", train_index, "TEST:", test_index)
    ...     X_train, X_test = X[train_index], X[test_index]
    ...     y_train, y_test = y[train_index], y[test_index]
    ...
    TRAIN: [1 2] TEST: [0 3]
    TRAIN: [0 3] TEST: [1 2]
    TRAIN: [1 3] TEST: [0 2]
    TRAIN: [0 2] TEST: [1 3]
    Notes
    -----
    Randomized CV splitters may return different results for each call of
    split. You can make the results identical by setting `random_state`
    to an integer.
    See Also
    --------
    RepeatedKFold : Repeats K-Fold n times.
    """
    def __init__(self, *, n_splits: List[int] = [3, 5, 10, 5, 3], n_repeats: int = 1,
                 random_state: Optional[np.random.RandomState] = None):
        assert len(n_splits) == n_repeats, "Repetitions come through n_splits schedule"
        super().__init__(
            StratifiedKFold, n_repeats=n_repeats, random_state=random_state,
            n_splits=n_splits)


class RepeatedMultiKFold(_RepeatedMultiSplits):
    """Repeated Multi-K-Fold cross validator.
    Repeats Multi K-Fold n times with different randomization in each
    repetition, for multiple k-splits.
    Read more in the :ref:`User Guide <repeated_k_fold>`.
    Parameters
    ----------
    n_splits : List[int], default=[3, 5, 10]
        Number of folds. Must be at least 2.
    n_repeats : int, default=1
        Number of times cross-validator needs to be repeated.
    random_state : int, RandomState instance or None, default=None
        Controls the generation of the random states for each repetition.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import RepeatedMultiKFold
    >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    >>> y = np.array([0, 0, 1, 1])
    >>> rskf = RepeatedMultiKFold(n_splits=2, n_repeats=2,
    ...     random_state=36851234)
    >>> for train_index, test_index in rskf.split(X, y):
    ...     print("TRAIN:", train_index, "TEST:", test_index)
    ...     X_train, X_test = X[train_index], X[test_index]
    ...     y_train, y_test = y[train_index], y[test_index]
    ...
    TRAIN: [1 2] TEST: [0 3]
    TRAIN: [0 3] TEST: [1 2]
    TRAIN: [1 3] TEST: [0 2]
    TRAIN: [0 2] TEST: [1 3]
    Notes
    -----
    Randomized CV splitters may return different results for each call of
    split. You can make the results identical by setting `random_state`
    to an integer.
    See Also
    --------
    RepeatedKFold : Repeats K-Fold n times.
    """
    def __init__(self, *, n_splits: List[int] = [3, 5, 10, 5, 3], n_repeats: int = 1,
                 random_state: Optional[np.random.RandomState] = None):
        assert len(n_splits) == n_repeats, "Repetitions come through n_splits schedule"
        super().__init__(
            KFold, n_repeats=n_repeats, random_state=random_state,
            n_splits=n_splits)
