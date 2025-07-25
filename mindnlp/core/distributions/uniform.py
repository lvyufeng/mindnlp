# mypy: allow-untyped-defs
from mindnlp import core
from mindnlp.core import nan, Tensor
from . import constraints
from .distribution import Distribution
from .utils import broadcast_all
from mindnlp.core.types import _Number, _size


__all__ = ["Uniform"]


class Uniform(Distribution):
    r"""
    Generates uniformly distributed random samples from the half-open interval
    ``[low, high)``.

    Example::

        >>> m = Uniform(torch.tensor([0.0]), torch.tensor([5.0]))
        >>> m.sample()  # uniformly distributed in the range [0.0, 5.0)
        >>> # xdoctest: +SKIP
        tensor([ 2.3418])

    Args:
        low (float or Tensor): lower range (inclusive).
        high (float or Tensor): upper range (exclusive).
    """

    # TODO allow (loc,scale) parameterization to allow independent constraints.
    arg_constraints = {
        "low": constraints.dependent(is_discrete=False, event_dim=0),
        "high": constraints.dependent(is_discrete=False, event_dim=0),
    }
    has_rsample = True

    @property
    def mean(self) -> Tensor:
        return (self.high + self.low) / 2

    @property
    def mode(self) -> Tensor:
        return nan * self.high

    @property
    def stddev(self) -> Tensor:
        return (self.high - self.low) / 12**0.5

    @property
    def variance(self) -> Tensor:
        return (self.high - self.low).pow(2) / 12

    def __init__(self, low, high, validate_args=None):
        self.low, self.high = broadcast_all(low, high)

        if isinstance(low, _Number) and isinstance(high, _Number):
            batch_shape = core.Size()
        else:
            batch_shape = self.low.size()
        super().__init__(batch_shape, validate_args=validate_args)

        if self._validate_args and not core.lt(self.low, self.high).all():
            raise ValueError("Uniform is not defined when low>= high")

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Uniform, _instance)
        batch_shape = core.Size(batch_shape)
        new.low = self.low.expand(batch_shape)
        new.high = self.high.expand(batch_shape)
        super(Uniform, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return constraints.interval(self.low, self.high)

    def rsample(self, sample_shape: _size = core.Size()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        rand = core.rand(shape, dtype=self.low.dtype, device=self.low.device)
        return self.low + rand * (self.high - self.low)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        lb = self.low.le(value).type_as(self.low)
        ub = self.high.gt(value).type_as(self.low)
        return core.log(lb.mul(ub)) - core.log(self.high - self.low)

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        result = (value - self.low) / (self.high - self.low)
        return result.clamp(min=0, max=1)

    def icdf(self, value):
        result = value * (self.high - self.low) + self.low
        return result

    def entropy(self):
        return core.log(self.high - self.low)