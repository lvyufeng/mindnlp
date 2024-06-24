from .._operations.nn import raw_gelu, raw_relu, raw_softmax, raw_linear

def gelu(input, approximate='none'):
    return raw_gelu(input)

def relu(input):
    return raw_relu(input)

def softmax(input, dim):
    return raw_softmax(input, dim)

def linear(input, weight, bias):
    return raw_linear(input, weight, bias)
