import mindspore
from mindnlp.core.ops.array import raw_cast
from mindnlp.core import Tensor

import numpy as np

def test_cast_obj():
    x = np.random.randn(3, 4).astype(np.float32)
    x_tensor = Tensor(x)
    print(x_tensor)
    x_cast = raw_cast(x_tensor, mindspore.float16)
    assert isinstance(x_cast, Tensor)
    print(x_cast)
    x_cast = raw_cast(x_cast, mindspore.float32)
    assert isinstance(x_cast, Tensor)
    print(x_cast)