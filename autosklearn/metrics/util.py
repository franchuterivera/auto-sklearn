# -*- encoding: utf-8 -*-
import numpy as np
from autosklearn.util.common import print_getrusage


def sanitize_array(array):
    """
    Replace NaN and Inf (there should not be any!)
    :param array:
    :return:
    """
    print_getrusage(f"Sanitize array start")
    a = np.ravel(array)
    maxi = np.nanmax(a[np.isfinite(a)])
    mini = np.nanmin(a[np.isfinite(a)])
    array[array == float('inf')] = maxi
    array[array == float('-inf')] = mini
    mid = (maxi + mini) / 2
    array[np.isnan(array)] = mid
    print_getrusage(f"Sanitize array end")
    return array
