# -*- encoding: utf-8 -*-

import os
import warnings

from resource import *
from tabulate import tabulate

from memory_profiler import profile

__all__ = [
    'check_pid',
    'warn_if_not_float'
]


def print_getrusage(message):
    print("\nLocation: {}".format(message))
    for member in [RUSAGE_SELF, RUSAGE_CHILDREN, RUSAGE_BOTH, RUSAGE_THREAD]:
        print(f"member={member}")
        columns = ["Index", "Field", "Resource"]
        fields = [
            'ru_utime',
            'ru_stime',
            'ru_maxrss',
            'ru_ixrss',
            'ru_idrss',
            'ru_isrss',
            'ru_minflt',
            'ru_majflt',
            'ru_nswap',
            'ru_inblock',
            'ru_oublock',
            'ru_msgsnd',
            'ru_msgrcv',
            'ru_nsignals',
            'ru_nvcsw',
            'ru_nivcsw',
        ]

        rusage = getrusage(member)
        rows = []
        for i, field in enumerate(fields):
            rows.append([i, field, getattr(rusage, field)])
        print(tabulate(rows, columns, tablefmt="grid"))
        print("\n")


@profile
def warn_if_not_float(X, estimator='This algorithm'):
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


@profile
def check_pid(pid):
    """Check For the existence of a unix pid."""
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    else:
        return True


@profile
def check_true(p):
    if p in ("True", "true", 1, True):
        return True
    return False


@profile
def check_false(p):
    if p in ("False", "false", 0, False):
        return True
    return False


@profile
def check_none(p):
    if p in ("None", "none", None):
        return True
    return False


@profile
def check_for_bool(p):
    if check_false(p):
        return False
    elif check_true(p):
        return True
    else:
        raise ValueError("%s is not a bool" % str(p))
