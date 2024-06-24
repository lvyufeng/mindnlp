from mindspore import ops
from mindspore.ops import Primitive
from .executor import execute

cast_op = Primitive('Cast')
cast_op.init_prim_io_names(inputs=['x', 'dst_type'], outputs=['output'])
def raw_cast(x, dtype):
    return execute(cast_op, x, dtype)

_zeros = Primitive('Zeros')
def raw_zeros(shape, dtype):
    return execute(_zeros, shape, dtype)

stridedslice_op = Primitive('StridedSlice')
stridedslice_op.init_prim_io_names(inputs=['x', 'begin', 'end', 'strides'], outputs=['output'])
def raw_strided_slice(x, begin, end, strides, begin_mask=0, end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=0):
    stridedslice_op.add_prim_attr('begin_mask', begin_mask)
    stridedslice_op.add_prim_attr('end_mask', end_mask)
    stridedslice_op.add_prim_attr('ellipsis_mask', ellipsis_mask)
    stridedslice_op.add_prim_attr('new_axis_mask', new_axis_mask)
    stridedslice_op.add_prim_attr('shrink_axis_mask', shrink_axis_mask)
    return execute(stridedslice_op, x, begin, end, strides)

unsqueeze_op = Primitive('ExpandDims')
unsqueeze_op.init_prim_io_names(inputs=['x', 'axis'], outputs=['output'])
def raw_unsqueeze(x, axis):
    return execute(unsqueeze_op, x, axis)

squeeze_op = Primitive('Squeeze')
squeeze_op.init_prim_io_names(inputs=['x'], outputs=['output'])
def raw_squeeze(x, axis):
    if not isinstance(axis, tuple):
        axis = (axis,)
    squeeze_op.add_prim_attr("axis", axis)
    return execute(squeeze_op, x)

_reshape = ops.Reshape()
def raw_reshape(x, shape):
    return execute(_reshape, x, shape)

_transpose = ops.Transpose()
def raw_transpose(x, perm):
    return execute(_transpose, x, perm)
