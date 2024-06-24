import numpy as np
from mindspore import dtype_to_nptype
from .creation import *
from .math import *
from .nn import *
from .array import *

def finfo(dtype):
    return np.finfo(dtype_to_nptype(dtype))
