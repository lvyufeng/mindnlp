# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""mindnlp tensor"""
from copy import deepcopy
from mindspore import ops
from mindspore._c_expression import Tensor as MSTensor
from mindspore._c_expression import TensorNode
import mindnlp.core as core

class Tensor:
    tensor = None
    stub = None
    def __init__(self, input, dtype=None): # pylint: disable=super-init-not-called
        if isinstance(input, TensorNode):
            self.stub = input
        elif isinstance(input, MSTensor):
            self.tensor = input
        else:
            self.tensor = MSTensor(input, dtype=dtype)

    @property
    def data(self):
        if self.tensor is not None:
            return self.tensor
        return self.stub_sync()

    @property
    def shape(self):
        """shape stub."""
        if self.stub:
            if not hasattr(self, "stub_shape"):
                self.stub_shape = self.stub.get_shape()
            return self.stub_shape
        return tuple(self.tensor.shape)

    @property
    def ndim(self):
        return len(self.shape)

    def dim(self):
        return self.ndim

    @property
    def dtype(self):
        """dtype stub."""
        if self.stub:
            if not hasattr(self, "stub_dtype"):
                self.stub_dtype = self.stub.get_dtype()
            return self.stub_dtype
        return self.tensor.dtype

    def size(self, dim=None):
        if dim is None:
            return self.shape
        else:
            return self.shape[dim]

    def stub_sync(self):
        """sync real tensor."""
        if self.stub:
            val = self.stub.get_value()
            self.tensor = MSTensor(val)
            if hasattr(self, "member_cache"):
                for k, v in self.member_cache.items():
                    setattr(self.tensor, k, v)
            self.stub = None
        return self.tensor

    def __hash__(self):
        return hash(id(self))

    def copy_(self, value):
        if isinstance(value, Tensor):
            self.stub = value.stub
            self.tensor = value.tensor
        elif isinstance(value, MSTensor):
            self.stub = None
            self.tensor = value
        else:
            raise ValueError(f'not support type: {type(value)}')

    def set_data(self, data):
        self.copy_(data)
    
    def clone(self):
        return deepcopy(self)

    def __repr__(self) -> str:
        self.data.data_sync(True)
        return self.data.__repr__()
    
    def __getitem__(self, slices):
        slices = core._operations.utils.slice_helper(slices)
        return core._operations.array.raw_strided_slice(self, *slices)

    def __add__(self, other):
        return core.ops.add(self, other)

    def __radd__(self, other):
        return core.ops.add(other, self)

    def __truediv__ (self, other):
        return core.ops.div(self, other)

    def __rtruediv__ (self, other):
        return core.ops.rdiv(self, other)

    def broadcast_to(self, shape):
        return core._operations.other.raw_broadcast_to(self, shape)

    def expand(self, *size):
        if len(size) == 1:
            size = size[0]
        return self.broadcast_to(size)

    def reshape(self, *shape):
        return core.ops.reshape(self, *shape)

    def view(self, *shape):
        return self.reshape(*shape)
    
    def permute(self, *dims):
        return core.ops.permute(self, dims)
    
    def transpose(self, dim0, dim1):
        return core.ops.transpose(self, dim0, dim1)

    def flatten(self, start_dim=0, end_dim=-1):
        return core.ops.flatten(self, start_dim, end_dim)

    def unflatten(self, dim, sizes):
        return core.ops.unflatten(self, dim, sizes)

def tensor(data, *, dtype=None):
    return Tensor(data, dtype)
