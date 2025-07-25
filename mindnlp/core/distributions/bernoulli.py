"""bernoulli"""
# mypy: allow-untyped-defs
# pylint: disable=method-hidden
from numbers import Number

from math import nan
from . import constraints
from .exp_family import ExponentialFamily
from .utils import (
    broadcast_all,
    lazy_property,
    logits_to_probs,
    probs_to_logits,
)
from .. import ops
from ..autograd import no_grad
from ..nn.functional import binary_cross_entropy_with_logits


__all__ = ["Bernoulli"]


class Bernoulli(ExponentialFamily):
    r"""
    Creates a Bernoulli distribution parameterized by :attr:`probs`
    or :attr:`logits` (but not both).

    Samples are binary (0 or 1). They take the value `1` with probability `p`
    and `0` with probability `1 - p`.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Bernoulli(core.tensor([0.3]))
        >>> m.sample()  # 30% chance 1; 70% chance 0
        tensor([ 0.])

    Args:
        probs (Number, Tensor): the probability of sampling `1`
        logits (Number, Tensor): the log-odds of sampling `1`
    """
    arg_constraints = {"probs": constraints.unit_interval, "logits": constraints.real}
    support = constraints.boolean
    has_enumerate_support = True
    _mean_carrier_measure = 0

    def __init__(self, probs=None, logits=None, validate_args=None):
        if (probs is None) == (logits is None):
            raise ValueError(
                "Either `probs` or `logits` must be specified, but not both."
            )
        if probs is not None:
            is_scalar = isinstance(probs, Number)
            (self.probs,) = broadcast_all(probs)
        else:
            is_scalar = isinstance(logits, Number)
            (self.logits,) = broadcast_all(logits)
        self._param = self.probs if probs is not None else self.logits
        if is_scalar:
            batch_shape = ()
        else:
            batch_shape = self._param.shape
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Bernoulli, _instance)
        if "probs" in self.__dict__:
            new.probs = self.probs.broadcast_to(batch_shape)
            new._param = new.probs
        if "logits" in self.__dict__:
            new.logits = self.logits.broadcast_to(batch_shape)
            new._param = new.logits
        super(Bernoulli, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def _new(self, *args, **kwargs):
        return self._param.new(*args, **kwargs)

    @property
    def mean(self):
        return self.probs

    @property
    def mode(self):
        mode = (self.probs >= 0.5).to(self.probs)
        mode[self.probs == 0.5] = nan
        return mode

    @property
    def variance(self):
        return self.probs * (1 - self.probs)

    @lazy_property
    def logits(self):
        return probs_to_logits(self.probs, is_binary=True)

    @lazy_property
    def probs(self):
        return logits_to_probs(self.logits, is_binary=True)

    @property
    def param_shape(self):
        return self._param.shape

    def sample(self, sample_shape=()):
        shape = self._extended_shape(sample_shape)
        with no_grad():
            return ops.bernoulli(self.probs.broadcast_to(shape))

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        logits, value = broadcast_all(self.logits, value)
        return -binary_cross_entropy_with_logits(logits, value, reduction="none")

    def entropy(self):
        return binary_cross_entropy_with_logits(
            self.logits, self.probs, reduction="none"
        )

    def enumerate_support(self, expand=True):
        values = ops.arange(2, dtype=self._param.dtype)
        values = values.view((-1,) + (1,) * len(self._batch_shape))
        if expand:
            values = values.broadcast_to((-1,) + self._batch_shape)
        return values

    @property
    def _natural_params(self):
        return (ops.logit(self.probs),)

    def _log_normalizer(self, x):
        return ops.log1p(ops.exp(x))
