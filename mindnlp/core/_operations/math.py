import mindspore
from mindspore import ops
from mindspore.ops import Primitive
from .executor import execute
from .array import raw_squeeze, raw_unsqueeze

BACKEND = mindspore.get_context('device_target')


matmul_op = Primitive('MatMul')
matmul_op.init_prim_io_names(inputs=['x1', 'x2'], outputs=['output'])
def raw_matmul(x, y, transpose_a=False, transpose_b=False):
    matmul_op.add_prim_attr('transpose_a', transpose_a)
    matmul_op.add_prim_attr('transpose_b', transpose_b)
    if BACKEND == 'Ascend':
        matmul_op.add_prim_attr('transpose_x1', transpose_a)
        matmul_op.add_prim_attr('transpose_x2', transpose_b)

    if len(x.shape) == 1:
        x = raw_unsqueeze(x, 1 if transpose_a else 0)
        out = execute(matmul_op, x, y)
        out = raw_squeeze(out, 1 if transpose_a else 0)
        return out

    if len(y.shape) == 1:
        y = raw_unsqueeze(y, 0 if transpose_b else 1)
        out = execute(matmul_op, x, y)
        out = raw_squeeze(out, 0 if transpose_b else 1)
        return out

    return execute(matmul_op, x, y)

_batch_matmul = Primitive('BatchMatMul')
_batch_matmul.init_prim_io_names(inputs=['x1', 'x2'], outputs=['output'])
def raw_batch_matmul(x, y, transpose_a=False, transpose_b=False):
    _batch_matmul.add_prim_attr('transpose_a', transpose_a)
    _batch_matmul.add_prim_attr('transpose_b', transpose_b)
    if BACKEND == 'Ascend':
        _batch_matmul.add_prim_attr('adj_x1', transpose_a)
        _batch_matmul.add_prim_attr('adj_x2', transpose_b)

    return execute(_batch_matmul, x, y)

_addcmul = ops.Addcmul()
def raw_addcmul(input, tensor0, tensor1, value):
    return execute(_addcmul, input, tensor0, tensor1, value)

_addcdiv = ops.Addcdiv()
def raw_addcdiv(input, tensor0, tensor1, value):
    return execute(_addcdiv, input, tensor0, tensor1, value)

_add = ops.Add()
def raw_add(x, y):
    return execute(_add, x, y)

_sub = ops.Sub()
def raw_sub(x, y):
    return execute(_sub, x, y)

_mul = ops.Mul()
def raw_mul(x, y):
    return execute(_mul, x, y)

_div = ops.Div()
def raw_div(x, y):
    return execute(_div, x, y)

_pow = ops.Pow()
def raw_pow(x, pow):
    return execute(_pow, x, pow)

_sqrt = ops.Sqrt()
def raw_sqrt(x):
    return execute(_sqrt, x)
