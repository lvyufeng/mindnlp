
import functools
import mindspore
from .tensor import Tensor

def core_patch_decorator(patch_instance, func):
    @functools.wraps(func)
    def decorate_autocast(*args, **kwargs):
        with patch_instance:
            return func(*args, **kwargs)

    return decorate_autocast

class CorePatch:
    def __enter__(self):
        # Replace StubTensor with Tensor
        if mindspore.common._stub_tensor.StubTensor == Tensor:
            return self
        self.original_stub_tensor = mindspore.common._stub_tensor.StubTensor
        mindspore.common._stub_tensor.StubTensor = Tensor
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original StubTensor
        if mindspore.common._stub_tensor.StubTensor != Tensor:
            mindspore.common._stub_tensor.StubTensor = self.original_stub_tensor

    def __call__(self, func):
        return core_patch_decorator(self, func)