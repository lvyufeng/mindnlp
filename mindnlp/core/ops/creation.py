import mindspore
import numpy as np
from ..tensor import Tensor
from .._operations.array import raw_zeros

def arange(*args, dtype=None):
    return Tensor(np.arange(*args), dtype)

def zeros(*size, dtype=None):
    if isinstance(size[0], tuple):
        size = size[0]
    if dtype is None:
        dtype = mindspore.float32
    return raw_zeros(size, dtype)
