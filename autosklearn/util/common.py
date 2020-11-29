# -*- encoding: utf-8 -*-

import os
import warnings

import numpy as np

__all__ = [
    'check_pid',
    'warn_if_not_float'
]


def warn_if_not_float(X: np.ndarray, estimator: str = 'This algorithm') -> bool:
    """Warning utility function to check that data type is floating point.
    Returns True if a warning was raised (i.e. the input is not float) and
    False otherwise, for easier input validation.
    """
    if not isinstance(estimator, str):
        estimator = estimator.__class__.__name__
    if X.dtype.kind != 'f':
        warnings.warn("%s assumes floating point values as input, "
                      "got %s" % (estimator, X.dtype))
        return True
    return False


def check_pid(pid: int) -> bool:
    """Check For the existence of a unix pid."""
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    else:
        return True


def check_true(p: str) -> bool:
    if p in ("True", "true", 1, True):
        return True
    return False


def check_false(p: str) -> bool:
    if p in ("False", "false", 0, False):
        return True
    return False


def check_none(p: str) -> bool:
    if p in ("None", "none", None):
        return True
    return False


def check_for_bool(p: str) -> bool:
    if check_false(p):
        return False
    elif check_true(p):
        return True
    else:
        raise ValueError("%s is not a bool" % str(p))


def thresholdout(train_loss, holdout_loss, thresholdout_scale=0.1):
    """
    Given a function $\phi$ which is a validation statistic, the algorithm first checks if the difference between the average value of $\phi$ on the training set St and the average value of $\phi$ on the holdout set Sh is below a certain threshold T + η. Here, T is a fixed number such as 0.01, and η is a Laplace noise variable of standard deviation smaller than T by a small factor such as 4. If the difference is below the threshold, then the algorithm returns the expectation on $E_{S_{t}}$; that is, the value of $\phi$ on the training set. If the difference is above the threshold, then the algorithm returns the average value of the function on the holdout after adding Laplacian noise.
    """
    tol = thresholdout_scale / 4

    train_loss = np.array(train_loss)
    holdout_loss = np.array(holdout_loss)

    diffNoise = np.abs(train_loss - holdout_loss) - np.random.normal(0, tol, holdout_loss.shape)
    flipIdx = diffNoise > thresholdout_scale

    new_holdout_loss = np.copy(train_loss)
    new_holdout_loss[flipIdx] = np.copy(holdout_loss)[flipIdx] + np.random.normal(0, tol, new_holdout_loss[flipIdx].shape)
    return new_holdout_loss.item(0)

