from .._operations.array import raw_reshape, raw_transpose

def reshape(input, *shape):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]
    return raw_reshape(input, shape)

def permute(input, dims):
    return raw_transpose(input, dims)

def transpose(input, dim0, dim1):
    axes = list(range(input.ndim))
    if dim0 < 0:
        dim0 = input.ndim + dim0
    if dim1 < 0:
        dim1 = input.ndim + dim1
    axes[dim0] = dim1
    axes[dim1] = dim0
    return permute(input, tuple(axes))

def flatten(x, start_dim=1, end_dim=-1):
    """Flattens the input. Does not affect the batch size."""
    if end_dim < 0:
        end_dim = x.ndim + end_dim
    new_shape = x.shape[:start_dim] + (-1,) + x.shape[end_dim + 1:]
    return reshape(x, new_shape)

def unflatten(x, dim, sizes):
    new_shape = x.shape[:dim] + sizes
    return reshape(x, new_shape)