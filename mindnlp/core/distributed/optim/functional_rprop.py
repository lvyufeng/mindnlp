# mypy: allow-untyped-defs
from typing import Dict, List, Optional, Tuple

from mindnlp import core
from mindnlp import core.optim._functional as F
from mindnlp.core import Tensor
from core.distributed.optim._deprecation_warning import (
    _scripted_functional_optimizer_deprecation_warning,
)


__all__: List[str] = []


# Define a TorchScript compatible Functional Rprop Optimizer
# where we use these optimizer in a functional way.
# Instead of using the `param.grad` when updating parameters,
# we explicitly allow the distributed optimizer pass gradients to
# the `step` function. In this way, we could separate the gradients
# and parameters and allow multithreaded trainer to update the
# parameters without data traces on accumulating to the same .grad.
# NOTE: This should be only used by distributed optimizer internals
# and not meant to expose to the user.
@core.jit.script
class _FunctionalRprop:
    def __init__(
        self,
        params: List[Tensor],
        lr: float = 1e-2,
        etas: Tuple[float, float] = (0.5, 1.2),
        step_sizes: Tuple[float, float] = (1e-6, 50),
        foreach: bool = False,
        maximize: bool = False,
        _allow_empty_param_list: bool = False,
    ):
        _scripted_functional_optimizer_deprecation_warning(stacklevel=2)
        self.defaults = {
            "lr": lr,
        }
        self.etas = etas
        self.step_sizes = step_sizes
        self.foreach = foreach
        self.maximize = maximize

        if len(params) == 0 and not _allow_empty_param_list:
            raise ValueError("optimizer got an empty parameter list")

        # NOTE: we only have one param_group and don't allow user to add additional
        # param group as it's not a common use case.
        self.param_group = {"params": params}

        self.state = core.jit.annotate(Dict[core.Tensor, Dict[str, core.Tensor]], {})

    def step(self, gradients: List[Optional[Tensor]]):
        params = self.param_group["params"]
        params_with_grad = []
        grads = []
        prevs = []
        step_sizes = []
        state_steps = []
        lr = self.defaults["lr"]
        etaminus, etaplus = self.etas
        step_size_min, step_size_max = self.step_sizes

        if len(params) != len(gradients):
            raise ValueError(
                "the gradients passed in does not equal to the size of the parameters!"
                + f"Params length: {len(params)}. "
                + f"Gradients length: {len(gradients)}"
            )

        has_complex = False
        for param, gradient in zip(params, gradients):
            if gradient is not None:
                has_complex |= core.is_complex(param)
                params_with_grad.append(param)
                grads.append(gradient)
                # Lazy state initialization
                if param not in self.state:
                    self.state[param] = {}
                    state = self.state[param]
                    state["step"] = core.tensor(0.0)
                    state["prev"] = core.zeros_like(
                        param, memory_format=core.preserve_format
                    )
                    state["step_size"] = core.full_like(gradient, lr)

                state = self.state[param]
                prevs.append(state["prev"])
                step_sizes.append(state["step_size"])
                state_steps.append(state["step"])

        with core.no_grad():
            F.rprop(
                params_with_grad,
                grads,
                prevs,
                step_sizes,
                state_steps,
                step_size_min=step_size_min,
                step_size_max=step_size_max,
                etaminus=etaminus,
                etaplus=etaplus,
                foreach=self.foreach,
                maximize=self.maximize,
                has_complex=has_complex,
            )
