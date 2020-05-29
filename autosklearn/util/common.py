# -*- encoding: utf-8 -*-

import os
import warnings

from resource import *
from tabulate import tabulate

from memory_profiler import profile
from datetime import datetime
import sys
import os
import resource
import re
import psutil

__all__ = [
    'check_pid',
    'warn_if_not_float'
]


def print_getrusage(message):

    #RLIMIT_AS. The maximum size of the process's virtual memory (address space) in bytes.


    #RSS is the Resident Set Size and is used to show how much memory is allocated to that process and is in RAM. It does not include memory that is swapped out. It does include memory from shared libraries as long as the pages from those libraries are actually in memory. It does include all stack and heap memory.
    #
    #VSZ is the Virtual Memory Size. It includes all memory that the process can access, including memory that is swapped out, memory that is allocated, but not used, and memory that is from shared libraries.
    #
    #So if process A has a 500K binary and is linked to 2500K of shared libraries, has 200K of stack/heap allocations of which 100K is actually in memory (rest is swapped or unused), and it has only actually loaded 1000K of the shared libraries and 400K of its own binary then:
    #
    #RSS: 400K + 1000K + 100K = 1500K
    #VSZ: 500K + 2500K + 200K = 3200K

    print("\nLocation: {}".format(message))
    #Getting virtual memory size
    pid = os.getpid()
    with open(os.path.join("/proc", str(pid), "status")) as f:
        lines = f.readlines()
    _vmsize = [l for l in lines if l.startswith("VmSize")][0]
    vmsize = int(_vmsize.split()[1])
    print(f"vmsize={vmsize}")

    for key, value in dict(psutil.virtual_memory()._asdict()).items():
        print(f"psutil {key} = {value}")

    for member in [RUSAGE_SELF, RUSAGE_CHILDREN, RUSAGE_THREAD]:
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
