import torch
from typing import Iterable, List, NamedTuple, Tuple, Union
import numpy as np
from torch import Tensor
from ..._dtype import np2dtype, dtype2np


def _pad_sequence(sequences: List[torch.Tensor], batch_first: bool = True, padding_value: float = 0.0, padding_side: str = "right") -> torch.Tensor:
    """
    Pads a list of variable-length sequences to the same length using NumPy, mimicking PyTorch's pad_sequence.

    Args:
        sequences (List[torch.Tensor]): List of sequences (Tensors) to pad.
        batch_first (bool): If True, output shape will be (batch_size, max_len, *dims).
                            If False, output shape will be (max_len, batch_size, *dims).
        padding_value (float): The value used for padding.
        padding_side (str): Either 'left' or 'right', specifying where padding is applied.

    Returns:
        torch.Tensor: A tensor with padded sequences.
    """
    # Ensure valid padding_side input
    assert padding_side in ["left", "right"], "padding_side must be 'left' or 'right'"

    # Get the size of the sequences list
    sequences_size = len(sequences)

    # Get the max length of the sequences
    max_len = max([seq.size(0) for seq in sequences])

    # Get the trailing dimensions (if any)
    trailing_dims = sequences[0].size()[1:]

    # Create the padded tensor with the padding_value
    if batch_first:
        out_dims = (sequences_size, max_len) + trailing_dims
    else:
        out_dims = (max_len, sequences_size) + trailing_dims

    # Use the dtype of the first sequence to ensure consistency
    dtype = sequences[0].dtype
    dtype = dtype2np[dtype]
    out = np.full(out_dims, padding_value, dtype=dtype)  # Use the same dtype as input

    # Pad the sequences
    for i, seq in enumerate(sequences):
        length_i = seq.size(0)
        start = max_len - length_i if padding_side == "left" else 0

        # Convert to NumPy array and handle padding
        np_out = out[i] if batch_first else out[:, i]
        np_seq = seq.numpy()  # Convert to NumPy array

        if batch_first:
            np_out[start:start + length_i] = np_seq
        else:
            np_out[start:start + length_i] = np_seq

    # Convert the NumPy result back to a PyTorch tensor
    return torch.tensor(out, dtype=np2dtype[dtype])  # Ensure the output tensor has the same dtype as input

def pad_sequence(
    sequences: Union[Tensor, List[Tensor]],
    batch_first: bool = False,
    padding_value: float = 0.0,
) -> Tensor:
    r"""Pad a list of variable length Tensors with ``padding_value``

    ``pad_sequence`` stacks a list of Tensors along a new dimension,
    and pads them to equal length. For example, if the input is a list of
    sequences with size ``L x *`` and ``batch_first`` is False, the output is
    of size ``T x B x *``.

    `B` is batch size. It is equal to the number of elements in ``sequences``.
    `T` is length of the longest sequence.
    `L` is length of the sequence.
    `*` is any number of trailing dimensions, including none.

    Example:
        >>> from torch.nn.utils.rnn import pad_sequence
        >>> a = torch.ones(25, 300)
        >>> b = torch.ones(22, 300)
        >>> c = torch.ones(15, 300)
        >>> pad_sequence([a, b, c]).size()
        torch.Size([25, 3, 300])

    Note:
        This function returns a Tensor of size ``T x B x *`` or ``B x T x *``
        where `T` is the length of the longest sequence. This function assumes
        trailing dimensions and type of all the Tensors in sequences are same.

    Args:
        sequences (list[Tensor]): list of variable length sequences.
        batch_first (bool, optional): output will be in ``B x T x *`` if True, or in
            ``T x B x *`` otherwise. Default: False.
        padding_value (float, optional): value for padded elements. Default: 0.

    Returns:
        Tensor of size ``T x B x *`` if :attr:`batch_first` is ``False``.
        Tensor of size ``B x T x *`` otherwise
    """

    if not (torch.jit.is_tracing() or torch.jit.is_scripting()):
        # JIT doesn't support `Iterable`
        if not isinstance(sequences, Iterable):
            msg = ('pad_sequence: Expected iterable for input sequences, but got arg of type: '
                   f'{type(sequences)}')
            raise RuntimeError(msg)

        # In JIT context this leads to,
        # RuntimeError: cannot statically infer the expected size of a list in this context
        sequences = tuple(sequences)
    else:
        # For JIT, we only support Union[Tensor, Tuple[Tensor]]
        if isinstance(sequences, torch.Tensor):
            sequences = sequences.unbind(0)

    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    return _pad_sequence(sequences, batch_first, padding_value)
