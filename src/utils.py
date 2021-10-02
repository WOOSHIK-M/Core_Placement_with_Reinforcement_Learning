import time
import os

import numpy as np

from typing import Dict, Any
from runpy import run_path

from data.toy_lut import get_toy_lut


##########################
#### HELPER FUNCTIONs ####
##########################

def logging_time(original_fn):
    """Logging execution time."""
    def wrapper_fn(*args, **kwargs):
        """Wrap a function to be recorded execution time."""
        start_time = time.time()
        result = original_fn(*args, **kwargs)
        end_time = time.time()
        print("WorkingTime[{}]: {} s".format(original_fn.__name__, (end_time - start_time)), "\n")
        return result
    return wrapper_fn


def print_title(title):
    """Print out strings in the speicific format."""
    print('%s' % '=' * 40)
    print('{0:^40}'.format(title))
    print('%s' % '=' * 40 + '\n')
