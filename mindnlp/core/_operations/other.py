from mindspore import ops
from mindspore.ops import Primitive
from .executor import execute

_broadcast_to = Primitive('BroadcastTo')
def raw_broadcast_to(x, shape):
    _broadcast_to.add_prim_attr("shape", shape)
    return execute(_broadcast_to, x)
