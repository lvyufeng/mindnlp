"""fft"""
from mindspore import ops
from mindspore.ops._primitive_cache import _get_cache_prim
from ..configs import use_pyboost
from ..ops import narrow, roll
from ..nn import functional as F

def rfft(input, n=None, dim=-1, norm="backward"):
    if use_pyboost():
        return ops.rfft(input, n, dim, norm)
    if input.shape[dim] < n:
        pad_inf = (0, n - input.shape[dim])
        pad_dims = (0, 0) * (input.ndim - (dim + 1)) + pad_inf
        input = F.pad(input, pad_dims)
    else:
        input = narrow(input, dim, 0, n)
    _rfft = _get_cache_prim(ops.FFTWithSize)(input.ndim, False, True, norm)
    return _rfft(input)

def irfft(input, n=None, dim=-1, norm="backward"):
    if use_pyboost():
        return ops.irfft(input, n, dim, norm)
    if input.shape[dim] < n:
        pad_inf = (0, n - input.shape[dim])
        pad_dims = (0, 0) * (input.ndim - (dim + 1)) + pad_inf
        input = pad(input, pad_dims)
    else:
        input = narrow(input, dim, 0, n)
    _irfft = _get_cache_prim(ops.FFTWithSize)(input.ndim, True, True, norm)
    return _irfft(input)

def fftn(input, s=None, dim=None, norm=None):
    return ops.fftn(input, s, dim, norm)

def fft(input, s=None, dim=-1, norm=None):
    return ops.fft(input, s, dim, norm)

def fftshift(x, dim=None):
    return ops.fftshift(x, dim)

def ifftn(input, s=None, dim=None, norm=None, *, out=None):
    return ops.ifftn(input, s, dim, norm)

def ifftshift(input, dim=None):
    return ops.ifftshift(input, dim)

def fft2(input, s=None, dim=(-2, -1), norm=None):
    return ops.fft2(input, s, dim, norm)

def ifft2(input, s=None, dim=(-2, -1), norm=None):
    return ops.ifft2(input, s, dim, norm)

__all__ = ['fft', 'fftn', 'irfft', 'rfft']
