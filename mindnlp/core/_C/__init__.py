from mindspore import default_generator, Generator as msGenerator

from . import _nn

def _jit_set_profiling_executor(mode):
    pass

def _jit_set_profiling_executor(mode):
    pass

def _jit_set_profiling_mode(mode):
    pass

def _jit_override_can_fuse_on_cpu(mode):
    pass

def _jit_override_can_fuse_on_gpu(mode):
    pass

def _jit_set_texpr_fuser_enabled(mode):
    pass

def _debug_set_autodiff_subgraph_inlining(mode):
    pass

Graph = None
Value = None

DisableTorchFunctionSubclass = None

class Generator(msGenerator):
    def __init__(self, device='cpu'):
        super().__init__()
