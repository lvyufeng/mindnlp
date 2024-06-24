from .._operations.math import raw_matmul, raw_batch_matmul, raw_add, raw_div, raw_sub, raw_mul

def matmul(input, other):
    if input.ndim == 2 and other.ndim == 2:
        return raw_matmul(input, other)
    return raw_batch_matmul(input, other)

def add(input, other, *, alpha=1):
    if alpha == 1:
        return raw_add(input, other)
    other = other * alpha
    return raw_add(input, other)

def mul(x0, x1):
    return raw_mul(x0, x1)

def sub(x0, x1):
    return raw_sub(x0, x1)

def rsub(x0, x1):
    return sub(x1, x0)

def div(x0, x1):
    return raw_div(x0, x1)

def rdiv(x0, x1):
    return div(x1, x0)
