from copy import deepcopy
from datetime import timedelta
from functools import partial, wraps
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Type, Union

from mindnlp import core
from mindnlp import core.distributed as dist
from mindnlp.core import nn, optim
from core._guards import active_fake_mode
from core.distributed._composable.fsdp import FSDPModule
from core.distributed._composable.fsdp._fsdp_param_group import FSDPParamGroup
from core.distributed._tools.mem_tracker import _RefType, _State, MemTracker
from core.distributed.distributed_c10d import (
    _IllegalWork,
    ProcessGroup,
    ReduceOp,
    Work,
)
from core.futures import Future
from core.utils._python_dispatch import TorchDispatchMode
from core.utils._pytree import tree_map_only
from core.utils.weak import WeakIdKeyDictionary, weakref


_TOTAL_KEY = "Total"

__all__ = ["FSDPMemTracker"]


class _FSDPRefType(_RefType):
    """
    Enumerates categories of memory usage in FSDP modules, including parameters, gradients, activations,
    and optimizer states.

    Attributes:
        SHARDED_PARAM (str): Memory usage of sharded parameters.
        UNSHARDED_PARAM (str): Memory usage of unsharded parameters.
        SHARDED_GRAD (str): Memory usage of sharded gradients corresponding to the sharded parameters.
        UNSHARDED_GRAD (str): Memory usage of unsharded gradients corresponding to the unsharded parameters.
        ACT (str): Memory usage of activations and tensors from forward and AC recomputation.
        TEMP (str): Memory usage of temporary tensors during the backward pass including gradients of activations.
        ALL_GATHER (str): Memory usage of all_gather output tensor.
        REDUCE_SCATTER (str): Memory usage of reduce_scatter input tensor.
        OPT (str): Memory usage of tensors storing optimizer states.
        INP (str): Memory usage of input tensors.
    """

    SHARDED_PARAM = "Sharded Param"
    UNSHARDED_PARAM = "Unsharded Param"
    BUFFER = "Buffer"
    SHARDED_GRAD = "Sharded Grad"
    UNSHARDED_GRAD = "Unsharded Grad"
    ACT = "Activation"
    TEMP = "Temp"
    ALL_GATHER = "All Gather"
    REDUCE_SCATTER = "Reduce Scatter"
    OPT = "OptState"
    INP = "Inputs"


class _SavedFSDPMethods(NamedTuple):
    pre_backward: Callable
    post_backward: Callable


class _SavedCollectives(NamedTuple):
    all_gather_into_tensor: Callable
    reduce_scatter_tensor: Callable
    all_reduce: Callable
    barrier: Callable


class _FSDPModState(_State):
    """
    Enumerates the states of FSDP modules during the forward and backward passes.
    """

    BEF_PRE_FW = "Before Pre-Forward"
    AFT_PRE_FW = "After Pre-Forward"
    BEF_POST_FW = "Before Post-Forward"
    AFT_POST_FW = "After Post-Forward"
    BEF_PRE_BW = "Before Pre-Backward"
    AFT_PRE_BW = "After Pre-Backward"
    BEF_POST_BW = "Before Post-Backward"
    AFT_POST_BW = "After Post-Backward"
    PRE_FW_AC = "Pre-Forward AC"
    POST_FW_AC = "Post-Forward AC"
    PEAK_FW = "Peak Forward"
    PEAK_BW = "Peak Backward"


class _FSDPModMemStats:
    """
    A class to store the memory statistics of an FSDP module.

    Args:
        mod_fqn (str): The fully qualified name of the FSDP module.

    Attributes:
        snapshots (Dict[_FSDPModState, Dict[core.device, Dict[str, int]]]): A dictionary of memory snapshots
        of the module at different states as defined by ``_FSDPModState``. Each key is a device, and
        each value is another dictionary with keys as memory reference types defined by ``_FSDPRefType`` and
        values as the memory consumed in bytes.

    """

    def __init__(self, mod_fqn: str) -> None:
        self.mod_fqn = mod_fqn
        self.local_peak: Dict[core.device, int] = {}
        self.snapshots: Dict[
            _FSDPModState, List[Dict[core.device, Dict[str, int]]]
        ] = {}


class FSDPMemTracker(MemTracker):
    """
    A ``TorchDispatchMode`` based context manager that extends ``core.distributed._tools.mem_tracker.MemTracker`` to track
    and categorize the peak memory and module-wise memory usage of FSDP modules.

    It tracks the peak memory usage across all the devices of all the FSDP modules in the module tree and categorizes
    the tensor memory usage as defined by ``_FSDPRefType``. Further, it captures memory `snapshots` at different stages of
    the module execution defined by ``_FSDPModState``.

    Attributes:
        memory_tracking: A weakref key dictionary to store the memory statistics of each module. Each key is a reference
        to a module, and each value is a ``_FSDPModMemStats`` object that stores the memory statistics of the module.

    Args:
        mod (core.nn.Module): The root FSDP module to be tracked.
        optm (core.optim.Optimizer, optional): The optimizer to be tracked.

    Note: Please refer to ``core.distributed._tools.mem_tracker.MemTracker`` to learn about the limitations.

    Example usage

    .. code-block:: python

        module = ...
        optimizer = ...
        inp = ...
        fmt = FSDPMemTracker(module, optimizer)
        fmt.track_inputs((inp,))
        with fmt:
            optimizer.zero_grad()
            loss = module(inp)
            print("After Forward:")
            fmt.display_snapshot("current")
            loss.backward()
            optimizer.step()
        fmt.display_snapshot("peak")
        fmt.display_modulewise_snapshots(depth = 3, units = "MB")

    """

    def __init__(
        self,
        mod: core.nn.Module,
        optm: Optional[core.optim.Optimizer] = None,
    ) -> None:
        super().__init__()
        assert isinstance(mod, FSDPModule), "FSDPMemTracker only supports FSDP modules"
        self._root_mod = mod
        self._optm = optm
        self._in_fake_mode: bool = False
        self._fsdp_mod_to_saved_methods: WeakIdKeyDictionary = WeakIdKeyDictionary()
        self._saved_collectives: _SavedCollectives
        self._ref_class: Type[_RefType] = _FSDPRefType

    def _instrument_fsdp_sharded_params_grads(
        self, fsdp_param_group: FSDPParamGroup
    ) -> None:
        # Track sharded params and grads after initilization
        for fsdp_param in fsdp_param_group.fsdp_params:
            self._update_and_maybe_create_winfos(
                fsdp_param.sharded_param,
                _FSDPRefType.SHARDED_PARAM,
            )
            sharded_grad = fsdp_param.sharded_param.grad
            if sharded_grad is not None:
                self._update_and_maybe_create_winfos(
                    sharded_grad,
                    _FSDPRefType.SHARDED_GRAD,
                )

    def _fsdp_state_pre_forward(
        self,
        fsdp_mod: FSDPModule,
        orig_fsdp_state_pre_fw: Callable,
    ) -> Callable:
        # We capture memory snapshots before and after ``FSDPState._pre_forward`` to attribute the `unsharded` params
        # and `all_gather` buffers.  There are three cases:
        # Case 1: If the module is not in the ``memory_tracking`` dictionary, create a new ``_FSDPModMemStats``
        #         instance for the module and add it to the ``memory_tracking`` dictionary.
        # Case 2: If the module is already in the ``memory_tracking`` dictionary and we are in backward, this means
        #         we are in the AC region. We check if this is the top most module in the AC region. If it is,
        #         we store a weak reference and set the flag ``_in_ac`` to True.
        # Case 3: If the module is already in the ``memory_tracking`` dictionary and we are in forward, this means
        #         this module is called for the second time. If it is a root module, that means we are in the next
        #         iteration and we error out. If it is not a root module, that means it's a submodule that is being
        #         used multiple times in the same iteration, which we allow and track.
        # For Case 1 and 3, we also initialiaze the ``local_peak`` and ``PEAK_FW`` snapshot for the module.
        # For Case 2 we only capture 1 snapshot after ``FSDPState._pre_forward`` runs because it is a no-op.
        @wraps(orig_fsdp_state_pre_fw)
        def inner(*args: Any, **kwargs: Any) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
            mod_fqn = self._mod_tracker.get_known_fqn(fsdp_mod)
            assert mod_fqn is not None
            if fsdp_mod not in self.memory_tracking:
                mod_stat = _FSDPModMemStats(mod_fqn)
                self.memory_tracking[fsdp_mod] = mod_stat
                snapshot = self.get_tracker_snapshot()
                mod_stat.local_peak = {
                    dev: dev_snap[_TOTAL_KEY] for dev, dev_snap in snapshot.items()
                }
                mod_stat.snapshots.setdefault(_FSDPModState.PEAK_FW, []).append(
                    snapshot
                )
                mod_stat.snapshots.setdefault(_FSDPModState.BEF_PRE_FW, []).append(
                    deepcopy(snapshot)
                )
            elif not self._mod_tracker.is_bw:
                parents = self._mod_tracker.parents - {mod_fqn}
                if len(parents) == 1 and "Global" in parents:
                    raise NotImplementedError(
                        "FSDPMemTracker does not support memory tracking for multiple iterative calls."
                        " Either use ``reset_mod_stats`` to clear module memory stats for the previous iteration"
                        " or file a github issue if you need this feature."
                    )

            args, kwargs = orig_fsdp_state_pre_fw(*args, **kwargs)

            fsdp_state = fsdp_mod._get_fsdp_state()
            if fsdp_param_group := fsdp_state._fsdp_param_group:
                for fsdp_param in fsdp_param_group.fsdp_params:
                    self._update_and_maybe_create_winfos(
                        fsdp_param.unsharded_param,
                        _FSDPRefType.UNSHARDED_PARAM,
                    )
            mod_stat = self.memory_tracking[fsdp_mod]
            if self._mod_tracker.is_bw:
                state = _FSDPModState.PRE_FW_AC
                if self._ac_mod is None:
                    self._ac_mod = weakref.ref(fsdp_mod)
                    self._in_ac = True
            else:
                state = _FSDPModState.AFT_PRE_FW
            mod_stat.snapshots.setdefault(state, []).append(self.get_tracker_snapshot())
            return args, kwargs

        return inner

    def _fsdp_state_post_forward(
        self,
        fsdp_mod: FSDPModule,
        orig_fsdp_state_post_fw: Callable,
    ) -> Callable:
        # We capture memory snapshots before and after ``FSDPState._post_forward`` to capture the resharded state
        # if ``reshard_after_forward`` is not ``False``. There are two cases:
        # Case 1: This is called in backward, which means we are in the AC region. If this is the top most module
        #         in the AC region, we set the flag ``_in_ac`` to False.
        # Case 2: This is called in forward.
        @wraps(orig_fsdp_state_post_fw)
        def inner(*args: Any, **kwargs: Any) -> Any:
            mod_stat = self.memory_tracking[fsdp_mod]
            if self._mod_tracker.is_bw:
                state = _FSDPModState.POST_FW_AC
                if self._ac_mod is not None and self._ac_mod() is fsdp_mod:
                    self._ac_mod = None
                    self._in_ac = False
            else:
                state = _FSDPModState.BEF_POST_FW
            mod_stat.snapshots.setdefault(state, []).append(self.get_tracker_snapshot())

            output = orig_fsdp_state_post_fw(*args, **kwargs)

            if not self._mod_tracker.is_bw:
                mod_stat.snapshots.setdefault(_FSDPModState.AFT_POST_FW, []).append(
                    self.get_tracker_snapshot()
                )
            return output

        return inner

    def _fsdp_param_group_pre_backward(
        self,
        fsdp_mod: FSDPModule,
        orig_fsdp_param_group_pre_backward: Callable,
    ) -> Callable:
        # We capture memory snapshots before and after ``FSDPParamGroup.pre_backward`` to capture the pre-fetching
        # and unsharding of params. We also initialize ``local_peak`` and ``PEAK_BW`` snapshot for the module.
        @wraps(orig_fsdp_param_group_pre_backward)
        def inner(*args: Any, **kwargs: Any) -> None:
            mod_stat = self.memory_tracking[fsdp_mod]
            snapshot = self.get_tracker_snapshot()
            mod_stat.local_peak = {
                dev: dev_snap[_TOTAL_KEY] for dev, dev_snap in snapshot.items()
            }
            mod_stat.snapshots.setdefault(_FSDPModState.PEAK_BW, []).append(snapshot)
            mod_stat.snapshots.setdefault(_FSDPModState.BEF_PRE_BW, []).append(
                deepcopy(snapshot)
            )
            orig_fsdp_param_group_pre_backward(*args, **kwargs)

            mod_stat.snapshots.setdefault(_FSDPModState.AFT_PRE_BW, []).append(
                self.get_tracker_snapshot()
            )

        return inner

    def _fsdp_param_group_post_backward(
        self,
        fsdp_mod: FSDPModule,
        orig_fsdp_param_group_post_backward: Callable,
    ) -> Callable:
        # We capture the memory snapshots before and after ``FSDPParamGroup.post_backward`` to track and attribute
        # the `unsharded` grads before the post backward and then `sharded` grads and `reduce_scatter`  buffers
        # after the post backward.
        @wraps(orig_fsdp_param_group_post_backward)
        def inner(*args: Any, **kwargs: Any) -> None:
            fsdp_state = fsdp_mod._get_fsdp_state()
            if fsdp_param_group := fsdp_state._fsdp_param_group:
                for fsdp_param in fsdp_param_group.fsdp_params:
                    unsharded_grad = fsdp_param._unsharded_param.grad
                    if unsharded_grad is not None:
                        self._update_and_maybe_create_winfos(
                            unsharded_grad,
                            _FSDPRefType.UNSHARDED_GRAD,
                            update_existing=True,
                        )

            mod_stat = self.memory_tracking[fsdp_mod]
            mod_stat.snapshots.setdefault(_FSDPModState.BEF_POST_BW, []).append(
                self.get_tracker_snapshot()
            )

            orig_fsdp_param_group_post_backward(*args, **kwargs)

            if fsdp_param_group := fsdp_state._fsdp_param_group:
                for fsdp_param in fsdp_param_group.fsdp_params:
                    sharded_grad = fsdp_param.sharded_param.grad
                    if sharded_grad is not None:
                        self._update_and_maybe_create_winfos(
                            sharded_grad,
                            _FSDPRefType.SHARDED_GRAD,
                        )

            mod_stat.snapshots.setdefault(_FSDPModState.AFT_POST_BW, []).append(
                self.get_tracker_snapshot()
            )

        return inner

    def _instrument_fsdp_module(self) -> None:
        # We uninstall the existing `FSDPState._pre_forward` and `FSDPState._post_forward` hooks and install
        # our own hooks that wrap them. We choose this over monkey-patching `FSDPParamGroup.pre_forward` and
        # `FSDPParamGroup.post_forward` because during AC these won't be called.
        # TODO(@sanketpurandare): This will need to be modified after this PR (https://github.com/pytorch/pytorch/pull/127786)
        # lands. For backward we monkey-patch the `FSDPParamGroup.pre_backward` and `FSDPParamGroup.post_backward`.
        for module in self._root_mod.modules():
            if isinstance(module, FSDPModule):
                fsdp_state = module._get_fsdp_state()
                if fsdp_param_group := fsdp_state._fsdp_param_group:
                    self._instrument_fsdp_sharded_params_grads(fsdp_param_group)
                    fsdp_state._pre_forward_hook_handle.remove()
                    fsdp_state._post_forward_hook_handle.remove()
                    fsdp_state._pre_forward_hook_handle = (
                        module.register_forward_pre_hook(
                            self._fsdp_state_pre_forward(
                                module, fsdp_state._pre_forward
                            ),
                            prepend=True,
                            with_kwargs=True,
                        )
                    )
                    fsdp_state._post_forward_hook_handle = module.register_forward_hook(
                        self._fsdp_state_post_forward(module, fsdp_state._post_forward),
                        prepend=False,
                        always_call=True,
                    )
                    self._fsdp_mod_to_saved_methods[module] = _SavedFSDPMethods(
                        fsdp_param_group.pre_backward,
                        fsdp_param_group.post_backward,
                    )
                    fsdp_param_group.pre_backward = self._fsdp_param_group_pre_backward(  # type: ignore[assignment]
                        module, fsdp_param_group.pre_backward
                    )
                    fsdp_param_group.post_backward = (  # type: ignore[assignment]
                        self._fsdp_param_group_post_backward(
                            module, fsdp_param_group.post_backward
                        )
                    )

        for buffer in self._root_mod.buffers():
            self._update_and_maybe_create_winfos(
                buffer,
                _FSDPRefType.BUFFER,
            )

    def _instrument_optimizer(self) -> None:
        # Register a hook on the optimizer step to track the optimizer states.
        # The pre-hook is to set the flag ``_in_opt`` to True. The post-hook unsets the flag,
        # and also tracks any optimizer states that are created during the optimizer step.
        if self._optm is not None:
            self._track_optimizer_states(_FSDPRefType.OPT, self._optm)

            def _opt_step_pre_hook(
                optimizer: optim.Optimizer, args: Any, kwargs: Any
            ) -> None:
                self._in_opt = True

            def _opt_step_post_hook(
                optimizer: optim.Optimizer, args: Any, kwargs: Any
            ) -> None:
                self._track_optimizer_states(_FSDPRefType.OPT, optimizer)
                self._in_opt = False

            self._optimizer_hook_handles = (
                self._optm.register_step_pre_hook(_opt_step_pre_hook),
                self._optm.register_step_post_hook(_opt_step_post_hook),
            )

    def _register_module_and_optimizer_hooks(self) -> None:
        self._instrument_fsdp_module()
        self._instrument_optimizer()

    def _deregister_module_and_optimizer_hooks(self) -> None:
        for (
            fsdp_mod,
            saved_methods,
        ) in self._fsdp_mod_to_saved_methods.items():
            fsdp_state = fsdp_mod._get_fsdp_state()
            fsdp_state._pre_forward_hook_handle.remove()
            fsdp_state._post_forward_hook_handle.remove()
            fsdp_state._pre_forward_hook_handle = fsdp_mod.register_forward_pre_hook(
                fsdp_state._pre_forward, prepend=True, with_kwargs=True
            )
            fsdp_state._post_forward_hook_handle = fsdp_mod.register_forward_hook(
                fsdp_state._post_forward, prepend=False
            )
            if fsdp_param_group := fsdp_state._fsdp_param_group:
                fsdp_param_group.pre_backward = saved_methods.pre_backward
                fsdp_param_group.post_backward = saved_methods.post_backward
        self._fsdp_mod_to_saved_methods.clear()

        if self._optimizer_hook_handles is not None:
            for handle in self._optimizer_hook_handles:
                handle.remove()
            self._optimizer_hook_handles = None

    def _instrument_and_maybe_bypass_collectives(self) -> None:
        # Monkey-patching collectives is required because they do not work with `FakeTensorMode`
        # It's also easier to track `all_gather` and `reduce_scatter` buffers faithfully.
        self._saved_collectives = _SavedCollectives(
            dist.all_gather_into_tensor,
            dist.reduce_scatter_tensor,
            dist.all_reduce,
            dist.barrier,
        )

        class FakeWork(Work):
            def __init__(self) -> None:
                super().__init__()

            def get_future(self) -> Future:
                future: Future = Future()
                future.set_result(None)
                return future

            def wait(self, timeout: Optional[timedelta] = None) -> bool:
                return True

        @wraps(dist.all_gather_into_tensor)
        def all_gather_into_tensor(
            output_tensor: core.Tensor,
            input_tensor: core.Tensor,
            group: Union[ProcessGroup, None] = None,
            async_op: bool = False,
        ) -> Union[Work, _IllegalWork, None]:
            self._update_and_maybe_create_winfos(
                output_tensor,
                _FSDPRefType.ALL_GATHER,
                update_existing=True,
            )

            if self._in_fake_mode:
                if async_op:
                    return FakeWork()
                return None
            else:
                return self._saved_collectives.all_gather_into_tensor(
                    output_tensor, input_tensor, group, async_op
                )

        @wraps(dist.reduce_scatter_tensor)
        def reduce_scatter_tensor(
            output: core.Tensor,
            input: core.Tensor,
            op: ReduceOp.RedOpType = dist.ReduceOp.SUM,
            group: Union[ProcessGroup, None] = None,
            async_op: bool = False,
        ) -> Union[Work, _IllegalWork, None]:
            self._update_and_maybe_create_winfos(
                input,
                _FSDPRefType.REDUCE_SCATTER,
                update_existing=True,
            )

            if self._in_fake_mode:
                if async_op:
                    return FakeWork()
                return None
            else:
                return self._saved_collectives.reduce_scatter_tensor(
                    output, input, op, group, async_op
                )

        @wraps(dist.all_reduce)
        def all_reduce(
            tensor: core.Tensor,
            op: ReduceOp.RedOpType = dist.ReduceOp.SUM,
            group: Union[ProcessGroup, None] = None,
            async_op: bool = False,
        ) -> Union[Work, _IllegalWork, None]:
            if self._in_fake_mode:
                if async_op:
                    return FakeWork()
                return None
            else:
                return self._saved_collectives.all_reduce(tensor, op, group, async_op)

        @wraps(dist.barrier)
        def barrier(
            group: Union[ProcessGroup, None] = dist.GroupMember.WORLD,
            async_op: bool = False,
            device_ids: Union[List[int], None] = None,
        ) -> Union[Work, None]:
            if self._in_fake_mode:
                return None
            else:
                return self._saved_collectives.barrier(group, async_op, device_ids)

        dist.all_gather_into_tensor = all_gather_into_tensor
        dist.reduce_scatter_tensor = reduce_scatter_tensor
        dist.all_reduce = all_reduce
        dist.barrier = barrier

    def _restore_collectives(self) -> None:
        dist.all_gather_into_tensor = self._saved_collectives.all_gather_into_tensor
        dist.reduce_scatter_tensor = self._saved_collectives.reduce_scatter_tensor
        dist.all_reduce = self._saved_collectives.all_reduce
        dist.barrier = self._saved_collectives.barrier
        del self._saved_collectives

    def track_inputs(self, inputs: Tuple[Any, ...]) -> None:
        """
        This is used to track the input tensors to the model and annotate them as ``Inputs``.
        Args:
            inputs (Tuple[Any]): A tuple containing the input data. This can include tensors
                        as well as other data types. Only tensors will be tracked.
        """

        def _track_inputs(t: core.Tensor) -> None:
            self._update_and_maybe_create_winfos(
                t,
                _FSDPRefType.INP,
            )

        tree_map_only(core.Tensor, _track_inputs, inputs)

    def track_external(
        self, *external: Union[nn.Module, optim.Optimizer, core.Tensor]
    ) -> None:
        """This is no-op for ``FSDPMemTracker``"""

    def __enter__(self) -> "FSDPMemTracker":
        self._in_fake_mode = True if active_fake_mode() else False
        self._register_module_and_optimizer_hooks()
        self._instrument_and_maybe_bypass_collectives()
        self._track_resize()
        self._peak_mem_snap = self.get_tracker_snapshot()
        self._peak_mem = {
            dev: dev_snap[_TOTAL_KEY] for dev, dev_snap in self._peak_mem_snap.items()
        }
        self._mod_tracker.__enter__()
        TorchDispatchMode.__enter__(self)
        return self

    def __exit__(self, *args: Any) -> None:
        self._deregister_module_and_optimizer_hooks()
        self._restore_collectives()
        self._restore_resize()
        TorchDispatchMode.__exit__(self, *args)
        self._mod_tracker.__exit__(*args)

    def __torch_dispatch__(self, func, types, args=..., kwargs=None):  # type: ignore[no-untyped-def]
        res = func(*args, **kwargs or {})
        # If we are tracking an optimizer state, we use the optimizer reference type.
        # If we are in backward region and not in AC region, we use the backward reference type.
        # Else we use the forward reference type.
        if self._in_opt:
            reftype = _FSDPRefType.OPT
        elif self._mod_tracker.is_bw and not self._in_ac:
            reftype = _FSDPRefType.TEMP
        else:
            reftype = _FSDPRefType.ACT
        tree_map_only(core.Tensor, partial(self._track, reftype), res)
        peak_state = (
            _FSDPModState.PEAK_BW if self._mod_tracker.is_bw else _FSDPModState.PEAK_FW
        )
        self._update_peak_stats(peak_state)
        return res
