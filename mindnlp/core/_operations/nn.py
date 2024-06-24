from mindspore import ops
from mindspore.ops import Primitive
from mindspore.ops.operations._inner_ops import SiLU
from .executor import execute

_relu = Primitive('ReLU')
_relu.init_prim_io_names(inputs=['x'], outputs=['output'])
def raw_relu(x):
    return execute(_relu, x)

_gelu = Primitive('GeLU')
_gelu.init_prim_io_names(inputs=['x'], outputs=['output'])
def raw_gelu(x):
    return execute(_gelu, x,)

_silu = SiLU()
def raw_silu(x):
    return execute(_silu, x,)

_gelu_fast = ops.FastGeLU()
def raw_gelu_fast(x):
    return execute(_gelu_fast, x,)

_mish = ops.Mish()
def raw_mish(x):
    return execute(_mish, x,)

_relu6 = ops.ReLU6()


_softmax_crossentropy = Primitive('SparseSoftmaxCrossEntropyWithLogits')
_softmax_crossentropy.init_prim_io_names(inputs=['features', 'labels'], outputs=['output'])
_softmax_crossentropy.add_prim_attr('sens', 1.0)
def raw_softmax_crossentropy(logits, labels, grad=False):
    _softmax_crossentropy.add_prim_attr('is_grad', grad)
    return execute(_softmax_crossentropy, logits, labels)

_softmax_crossentropy_ascend = Primitive('SparseSoftmaxCrossEntropyWithLogitsV2')
_softmax_crossentropy_ascend.init_prim_io_names(inputs=['features', 'labels'], outputs=['loss', 'backprop'])
def raw_softmax_crossentropy_ascend(logits, labels):
    return execute(_softmax_crossentropy_ascend, logits, labels)

_conv2d = Primitive('Conv2D')
_conv2d.init_prim_io_names(inputs=['x', 'w'], outputs=['output'])
_conv2d.add_prim_attr('mode', 1) # only support mode=1
def raw_conv2d(x, w, out_channel, kernel_size, pad_mode="valid", pad=0, stride=1, dilation=1, groups=1, data_format="NCHW"):
    _conv2d.add_prim_attr("out_channel", out_channel)
    _conv2d.add_prim_attr("kernel_size", kernel_size)
    _conv2d.add_prim_attr("pad_mode", pad_mode)
    _conv2d.add_prim_attr("pad", pad)
    _conv2d.add_prim_attr('stride', stride)
    _conv2d.add_prim_attr('dilation', dilation)
    _conv2d.add_prim_attr('group', groups)
    _conv2d.add_prim_attr('groups', groups)
    _conv2d.add_prim_attr('data_format', data_format)
    return execute(_conv2d, x, w)

_bias_add = Primitive('BiasAdd')
_bias_add.init_prim_io_names(inputs=['x', 'b'], outputs=['output'])
_bias_add.add_prim_attr('data_format', 'NCHW')
def raw_bias_add(x, y):
    return execute(_bias_add, x, y)

_dropout = Primitive('Dropout')
_dropout.add_prim_attr('Seed0', 1)
_dropout.add_prim_attr('Seed1', 1)
def raw_dropout(x, dropout):
    _dropout.add_prim_attr('keep_prob', 1 - dropout)
    return execute(_dropout, 'Dropout', x)

_maxpool = Primitive('MaxPoolWithArgmaxV2')
_maxpool.init_prim_io_names(inputs=["x"], outputs=["output", "argmax"])
_maxpool.add_prim_attr("argmax_type", 4)
def raw_maxpool(x, kernel_size, strides=None, pads=0, dilation=(1, 1), ceil_mode=False):
    _maxpool.add_prim_attr("kernel_size", kernel_size)
    _maxpool.add_prim_attr("strides", strides)
    _maxpool.add_prim_attr("pads", pads)
    _maxpool.add_prim_attr("dilation", dilation)
    _maxpool.add_prim_attr("ceil_mode", ceil_mode)
    return execute(_maxpool, x)

_nll_loss = Primitive('NLLLoss')
_nll_loss.init_prim_io_names(inputs=['x', 'target', "weight"], outputs=['loss', 'total_weight'])
def raw_nll_loss(input, target, weight, ignore_index=-100, reduction='mean'):
    _nll_loss.add_prim_attr("ignore_index", ignore_index)
    _nll_loss.add_prim_attr("reduction", reduction)
    return execute(_nll_loss, input, target, weight)

_layer_norm = Primitive("LayerNorm")
def raw_layer_norm(input, weight, bias, begin_norm_axis=1, begin_params_axis=1, epsilon=1e-7):
    _layer_norm.add_prim_attr("begin_norm_axis", begin_norm_axis)
    _layer_norm.add_prim_attr("begin_params_axis", begin_params_axis)
    _layer_norm.add_prim_attr("epsilon", epsilon)

    return execute(_layer_norm, input, weight, bias)


_unfold = Primitive('Im2Col')
_unfold.init_prim_io_names(inputs=['x'], outputs=['y'])
def raw_unfold(x, ksizes, strides=1, dilations=1, pads=0):
    _unfold.add_prim_attr('ksizes', ksizes)
    _unfold.add_prim_attr('strides', strides)
    _unfold.add_prim_attr('dilations', dilations)
    _unfold.add_prim_attr('pads', pads)
    _unfold.add_prim_attr('padding_mode', "CALCULATED")
    return execute(_unfold, x)


_fold = Primitive('Col2Im')
_fold.init_prim_io_names(inputs=['x', 'output_size'], outputs=['y'])
def raw_fold(x, output_size, kernel_size, dilation=1, padding=0, stride=1):
    _fold.add_prim_attr('kernel_size', kernel_size)
    _fold.add_prim_attr('dilation', dilation)
    _fold.add_prim_attr('padding', padding)
    _fold.add_prim_attr('stride', stride)
    return execute(_fold, x)

_softmax = Primitive('Softmax')
_softmax.init_prim_io_names(inputs=['x'], outputs=['output'])
def raw_softmax(input, axis):
    if not isinstance(axis, tuple):
        axis = (axis,)
    _softmax.add_prim_attr('axis', axis)
    return execute(_softmax, input)

_linear = Primitive('Dense')
_linear.init_prim_io_names(inputs=['x', 'w', 'b'], outputs=["output"])
def raw_linear(x, w, b):
    if b is not None:
        _linear.add_prim_attr("has_bias", True)
    else:
        _linear.add_prim_attr("has_bias", False)
    return execute(_linear, x, w, b)

_softmax = Primitive('Softmax')
_softmax.init_prim_io_names(inputs=['x'], outputs=['output'])
def raw_softmax(input, axis):
    if not isinstance(axis, tuple):
        axis = (axis,)
    _softmax.add_prim_attr('axis', axis)
    return execute(_softmax, input)
