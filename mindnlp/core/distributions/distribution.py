"""distribution base class"""
# mypy: allow-untyped-defs
import warnings
import builtins
from typing import Any, Dict, Optional, Tuple, Union, List
from typing_extensions import deprecated

import mindspore
from .. import ops
from ..autograd import no_grad

from . import constraints
from .utils import lazy_property


__all__ = ["Distribution"]

_size = Union[List[builtins.int], Tuple[builtins.int, ...]]


class Distribution:
    r"""
    Distribution is the abstract base class for probability distributions.
    """

    has_rsample = False
    has_enumerate_support = False
    _validate_args = __debug__


    @staticmethod
    def set_default_validate_args(value: bool) -> None:
        """
        Sets whether validation is enabled or disabled.

        The default behavior mimics Python's ``assert`` statement: validation
        is on by default, but is disabled if Python is run in optimized mode
        (via ``python -O``). Validation may be expensive, so you may want to
        disable it once a model is working.

        Args:
            value (bool): Whether to enable validation.
        """
        if value not in [True, False]:
            raise ValueError
        Distribution._validate_args = value


    def __init__(
        self,
        batch_shape: tuple = (),
        event_shape: tuple = (),
        validate_args: Optional[bool] = None,
    ):
        self._batch_shape = batch_shape
        self._event_shape = event_shape
        if validate_args is not None:
            self._validate_args = validate_args
        if self._validate_args:
            try:
                arg_constraints = self.arg_constraints
            except NotImplementedError:
                arg_constraints = {}
                warnings.warn(
                    f"{self.__class__} does not define `arg_constraints`. "
                    + "Please set `arg_constraints = {}` or initialize the distribution "
                    + "with `validate_args=False` to turn off validation."
                )
            for param, constraint in arg_constraints.items():
                if constraints.is_dependent(constraint):
                    continue  # skip constraints that cannot be checked
                if param not in self.__dict__ and isinstance(
                    getattr(type(self), param), lazy_property
                ):
                    continue  # skip checking lazily-constructed args
                value = getattr(self, param)
                valid = constraint.check(value)
                if not valid.all():
                    raise ValueError(
                        f"Expected parameter {param} "
                        f"({type(value).__name__} of shape {tuple(value.shape)}) "
                        f"of distribution {repr(self)} "
                        f"to satisfy the constraint {repr(constraint)}, "
                        f"but found invalid values:\n{value}"
                    )
        super().__init__()


    def expand(self, batch_shape: tuple, _instance=None):
        """
        Returns a new distribution instance (or populates an existing instance
        provided by a derived class) with batch dimensions expanded to
        `batch_shape`. This method calls :class:`~mindspore.Tensor.expand` on
        the distribution's parameters. As such, this does not allocate new
        memory for the expanded distribution instance. Additionally,
        this does not repeat any args checking or parameter broadcasting in
        `__init__.py`, when an instance is first created.

        Args:
            batch_shape (tuple): the desired expanded size.
            _instance: new instance provided by subclasses that
                need to override `.expand`.

        Returns:
            New distribution instance with batch dimensions expanded to
            `batch_size`.
        """
        raise NotImplementedError


    @property
    def batch_shape(self) -> tuple:
        """
        Returns the shape over which parameters are batched.
        """
        return self._batch_shape

    @property
    def event_shape(self) -> tuple:
        """
        Returns the shape of a single sample (without batching).
        """
        return self._event_shape

    @property
    def arg_constraints(self) -> Dict[str, constraints.Constraint]:
        """
        Returns a dictionary from argument names to
        :class:`~core.distributions.constraints.Constraint` objects that
        should be satisfied by each argument of this distribution. Args that
        are not tensors need not appear in this dict.
        """
        raise NotImplementedError

    @property
    def support(self) -> Optional[Any]:
        """
        Returns a :class:`~core.distributions.constraints.Constraint` object
        representing this distribution's support.
        """
        raise NotImplementedError

    @property
    def mean(self) -> mindspore.Tensor:
        """
        Returns the mean of the distribution.
        """
        raise NotImplementedError

    @property
    def mode(self) -> mindspore.Tensor:
        """
        Returns the mode of the distribution.
        """
        raise NotImplementedError(f"{self.__class__} does not implement mode")

    @property
    def variance(self) -> mindspore.Tensor:
        """
        Returns the variance of the distribution.
        """
        raise NotImplementedError

    @property
    def stddev(self) -> mindspore.Tensor:
        """
        Returns the standard deviation of the distribution.
        """
        return self.variance.sqrt()


    def sample(self, sample_shape: tuple = ()) -> mindspore.Tensor:
        """
        Generates a sample_shape shaped sample or sample_shape shaped batch of
        samples if the distribution parameters are batched.
        """
        with no_grad():
            return self.rsample(sample_shape)



    def rsample(self, sample_shape: tuple = ()) -> mindspore.Tensor:
        """
        Generates a sample_shape shaped reparameterized sample or sample_shape
        shaped batch of reparameterized samples if the distribution parameters
        are batched.
        """
        raise NotImplementedError



    @deprecated(
        "`sample_n(n)` will be deprecated. Use `sample((n,))` instead.",
        category=FutureWarning,
    )
    def sample_n(self, n: int) -> mindspore.Tensor:
        """
        Generates n samples or n batches of samples if the distribution
        parameters are batched.
        """
        return self.sample(tuple((n,)))



    def log_prob(self, value: mindspore.Tensor) -> mindspore.Tensor:
        """
        Returns the log of the probability density/mass function evaluated at
        `value`.

        Args:
            value (Tensor):
        """
        raise NotImplementedError



    def cdf(self, value: mindspore.Tensor) -> mindspore.Tensor:
        """
        Returns the cumulative density/mass function evaluated at
        `value`.

        Args:
            value (Tensor):
        """
        raise NotImplementedError



    def icdf(self, value: mindspore.Tensor) -> mindspore.Tensor:
        """
        Returns the inverse cumulative density/mass function evaluated at
        `value`.

        Args:
            value (Tensor):
        """
        raise NotImplementedError



    def enumerate_support(self, expand: bool = True) -> mindspore.Tensor:
        """
        Returns tensor containing all values supported by a discrete
        distribution. The result will enumerate over dimension 0, so the shape
        of the result will be `(cardinality,) + batch_shape + event_shape`
        (where `event_shape = ()` for univariate distributions).

        Note that this enumerates over all batched tensors in lock-step
        `[[0, 0], [1, 1], ...]`. With `expand=False`, enumeration happens
        along dim 0, but with the remaining batch dimensions being
        singleton dimensions, `[[0], [1], ..`.

        To iterate over the full Cartesian product use
        `itertools.product(m.enumerate_support())`.

        Args:
            expand (bool): whether to expand the support over the
                batch dims to match the distribution's `batch_shape`.

        Returns:
            Tensor iterating over dimension 0.
        """
        raise NotImplementedError



    def entropy(self) -> mindspore.Tensor:
        """
        Returns entropy of distribution, batched over batch_shape.

        Returns:
            Tensor of shape batch_shape.
        """
        raise NotImplementedError



    def perplexity(self) -> mindspore.Tensor:
        """
        Returns perplexity of distribution, batched over batch_shape.

        Returns:
            Tensor of shape batch_shape.
        """
        return ops.exp(self.entropy())


    def _extended_shape(self, sample_shape: _size = ()) -> Tuple[int, ...]:
        """
        Returns the size of the sample returned by the distribution, given
        a `sample_shape`. Note, that the batch and event shapes of a distribution
        instance are fixed at the time of construction. If this is empty, the
        returned shape is upcast to (1,).

        Args:
            sample_shape (tuple): the size of the sample to be drawn.
        """
        if not isinstance(sample_shape, tuple):
            sample_shape = tuple(sample_shape)
        return tuple(sample_shape + self._batch_shape + self._event_shape)

    def _validate_sample(self, value: mindspore.Tensor) -> None:
        """
        Argument validation for distribution methods such as `log_prob`,
        `cdf` and `icdf`. The rightmost dimensions of a value to be
        scored via these methods must agree with the distribution's batch
        and event shapes.

        Args:
            value (Tensor): the tensor whose log probability is to be
                computed by the `log_prob` method.
        Raises
            ValueError: when the rightmost dimensions of `value` do not match the
                distribution's batch and event shapes.
        """
        if not isinstance(value, mindspore.Tensor):
            raise ValueError("The value argument to log_prob must be a Tensor")

        event_dim_start = len(value.shape) - len(self._event_shape)
        if value.shape[event_dim_start:] != self._event_shape:
            raise ValueError(
                f"The right-most size of value must match event_shape: {value.shape} vs {self._event_shape}."
            )

        actual_shape = value.shape
        expected_shape = self._batch_shape + self._event_shape
        for i, j in zip(reversed(actual_shape), reversed(expected_shape)):
            if i != 1 and j != 1 and i != j:
                raise ValueError(
                    f"Value is not broadcastable with batch_shape+event_shape: {actual_shape} vs {expected_shape}."
                )
        try:
            support = self.support
        except NotImplementedError:
            warnings.warn(
                f"{self.__class__} does not define `support` to enable "
                + "sample validation. Please initialize the distribution with "
                + "`validate_args=False` to turn off validation."
            )
            return
        assert support is not None
        valid = support.check(value)
        if not valid.all():
            raise ValueError(
                "Expected value argument "
                f"({type(value).__name__} of shape {tuple(value.shape)}) "
                f"to be within the support ({repr(support)}) "
                f"of the distribution {repr(self)}, "
                f"but found invalid values:\n{value}"
            )

    def _get_checked_instance(self, cls, _instance=None):
        if _instance is None and type(self).__init__ != cls.__init__:
            raise NotImplementedError(
                f"Subclass {self.__class__.__name__} of {cls.__name__} that defines a custom __init__ method "
                "must also define a custom .expand() method."
            )
        return self.__new__(type(self)) if _instance is None else _instance # pylint: disable=no-value-for-parameter

    def __repr__(self) -> str:
        param_names = [k for k, _ in self.arg_constraints.items() if k in self.__dict__]
        args_string = ", ".join(
            [
                f"{p}: {self.__dict__[p] if self.__dict__[p].numel() == 1 else self.__dict__[p].shape}"
                for p in param_names
            ]
        )
        return self.__class__.__name__ + "(" + args_string + ")"
