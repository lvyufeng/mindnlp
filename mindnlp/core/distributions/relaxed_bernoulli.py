# mypy: allow-untyped-defs
# pylint: disable=method-hidden
"""RelaxedBernoulli"""
from numbers import Number

from .. import ops
from . import constraints
from .distribution import Distribution
from .transformed_distribution import TransformedDistribution
from .transforms import SigmoidTransform
from .utils import (
    broadcast_all,
    clamp_probs,
    lazy_property,
    logits_to_probs,
    probs_to_logits,
)

__all__ = ["LogitRelaxedBernoulli", "RelaxedBernoulli"]



class LogitRelaxedBernoulli(Distribution):
    r"""
    Creates a LogitRelaxedBernoulli distribution parameterized by :attr:`probs`
    or :attr:`logits` (but not both), which is the logit of a RelaxedBernoulli
    distribution.

    Samples are logits of values in (0, 1). See [1] for more details.

    Args:
        temperature (Tensor): relaxation temperature
        probs (Number, Tensor): the probability of sampling `1`
        logits (Number, Tensor): the log-odds of sampling `1`

    [1] The Concrete Distribution: A Continuous Relaxation of Discrete Random
    Variables (Maddison et al., 2017)

    [2] Categorical Reparametrization with Gumbel-Softmax
    (Jang et al., 2017)
    """
    arg_constraints = {"probs": constraints.unit_interval, "logits": constraints.real}
    support = constraints.real

    def __init__(self, temperature, probs=None, logits=None, validate_args=None):
        self.temperature = temperature
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
        new = self._get_checked_instance(LogitRelaxedBernoulli, _instance)
        new.temperature = self.temperature
        if "probs" in self.__dict__:
            new.probs = self.probs.broadcast_to(batch_shape)
            new._param = new.probs
        if "logits" in self.__dict__:
            new.logits = self.logits.broadcast_to(batch_shape)
            new._param = new.logits
        super(LogitRelaxedBernoulli, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new


    def _new(self, *args, **kwargs):
        return self._param.new(*args, **kwargs)

    @lazy_property
    def logits(self):
        return probs_to_logits(self.probs, is_binary=True)

    @lazy_property
    def probs(self):
        return logits_to_probs(self.logits, is_binary=True)

    @property
    def param_shape(self):
        return self._param.shape


    def rsample(self, sample_shape=()):
        shape = self._extended_shape(sample_shape)
        probs = clamp_probs(self.probs.broadcast_to(shape))
        uniforms = clamp_probs(
            ops.rand(shape, dtype=probs.dtype)
        )
        return (
            uniforms.log() - (-uniforms).log1p() + probs.log() - (-probs).log1p()
        ) / self.temperature



    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        logits, value = broadcast_all(self.logits, value)
        diff = logits - value.mul(self.temperature)
        return self.temperature.log() + diff - 2 * diff.exp().log1p()


class RelaxedBernoulli(TransformedDistribution):
    r"""
    Creates a RelaxedBernoulli distribution, parametrized by
    :attr:`temperature`, and either :attr:`probs` or :attr:`logits`
    (but not both). This is a relaxed version of the `Bernoulli` distribution,
    so the values are in (0, 1), and has reparametrizable samples.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = RelaxedBernoulli(core.tensor([2.2]),
        ...                      core.tensor([0.1, 0.2, 0.3, 0.99]))
        >>> m.sample()
        tensor([ 0.2951,  0.3442,  0.8918,  0.9021])

    Args:
        temperature (Tensor): relaxation temperature
        probs (Number, Tensor): the probability of sampling `1`
        logits (Number, Tensor): the log-odds of sampling `1`
    """
    arg_constraints = {"probs": constraints.unit_interval, "logits": constraints.real}
    support = constraints.unit_interval
    has_rsample = True

    def __init__(self, temperature, probs=None, logits=None, validate_args=None):
        base_dist = LogitRelaxedBernoulli(temperature, probs, logits)
        super().__init__(base_dist, SigmoidTransform(), validate_args=validate_args)


    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(RelaxedBernoulli, _instance)
        return super().expand(batch_shape, _instance=new)


    @property
    def temperature(self):
        return self.base_dist.temperature

    @property
    def logits(self):
        return self.base_dist.logits

    @property
    def probs(self):
        return self.base_dist.probs
