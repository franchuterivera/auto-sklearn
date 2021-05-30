# -*- encoding: utf-8 -*-
import os
import resource
import warnings

import numpy as np

import psutil

__all__ = [
    'check_pid',
    'warn_if_not_float'
]


def print_memory(tag: str = '', extra: bool = True, include_all: bool = False) -> str:
    memory = []
    processes = [(str(os.getpid()), f"{tag}-current")]

    if include_all:
        processes.append((str(os.getppid()), f"{tag}-parent"))
        parent = psutil.Process(os.getpid())
        for children in parent.children(recursive=True):
            if children.pid:
                processes.append((str(children.pid), f"{tag}-children"))

    for pid, name in processes:
        filename = '/proc/' + str(pid) + '/status'
        if pid and os.path.exists('/proc/' + str(pid) + '/status'):
            with open(filename, 'r') as fin:
                data = fin.read()
                for line in data.split('\n'):
                    if 'Vm' not in line:
                        continue
                    data = data.strip().replace('\t', ' ')
                    memory.append(f"{name}-{pid}-{line}")
        memory.append("\n")

    if extra:
        memory.append(f"rsuage={resource.getrusage(resource.RUSAGE_SELF)}")

    return "\n".join(memory)


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
