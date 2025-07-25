# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
from typing import Any, Protocol, runtime_checkable

from mindnlp import core


@runtime_checkable
class _Checkpointable(Protocol):  # noqa: PYI046
    """
    Interface for checkpointable objects.
    Implemented as a protocol, implicit subtyping is supported so subclasses do not need to inherit this explicitly.
    This is to allow arbitrary objects/tensor subclasses to hook into DCP seamlessly through implementing the interface.
    """

    def __create_write_items__(self, fqn: str, object: Any):
        """
        Return a list of WriteItems based on object's contents.
        """
        raise NotImplementedError(
            "_Checkpointable._create_write_items is not implemented"
        )

    def __create_chunk_list__(self):
        """
        Return a list of `ChunkStorageMetadata` based on object's contents.
        """
        raise NotImplementedError(
            "_Checkpointable._create_chunk_list is not implemented"
        )

    def __get_tensor_shard__(self, index) -> core.Tensor:
        """
        Return a 'core.Tensor' shard based on 'MetadataIndex'.
        """
        raise NotImplementedError(
            "_Checkpointable._get_tensor_shard is not implemented"
        )
