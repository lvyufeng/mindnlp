# coding=utf-8
# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MindSpore TrOCR decoder model."""

import copy
import math
from typing import Optional, Tuple, Union

import numpy as np
import mindspore
from mindspore.common.initializer import initializer, Normal

from mindnlp.core import nn, ops
from mindnlp.core.nn import functional as F
from mindnlp.utils import logging

from ....common.activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, \
    _prepare_4d_causal_attention_mask
from ...modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, \
    CausalLMOutputWithCrossAttentions
from ...modeling_utils import PreTrainedModel
from .configuration_trocr import TrOCRConfig

logger = logging.get_logger(__name__)


class TrOCRLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # TrOCR is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, ids: mindspore.Tensor, past_key_values_length: int = 0):
        """`input_ids' shape is expected to be [bsz x seqlen]."""

        bsz, seq_len = ids.shape[:2]
        positions = ops.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=mindspore.int64
        ).broadcast_to((bsz, -1))

        return super().forward(positions + self.offset)


class TrOCRScaledWordEmbedding(nn.Embedding):
    """
    This module overrides nn.Embeddings' forward by multiplying with embeddings scale.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int,
                 embed_scale: Optional[float] = 1.0):
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.embed_scale = embed_scale

    def forward(self, ids: mindspore.Tensor):
        return super().forward(ids) * self.embed_scale


class TrOCRSinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None):
        super().__init__()
        self.offset = 2
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = self.get_embedding(num_positions, embedding_dim, padding_idx)
        self._float_tensor = mindspore.Tensor(1)

    @staticmethod
    def get_embedding(num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        """
        Build sinusoidal embeddings. This matches the implementation in tensor2tensor,
        but differs slightly from the
        description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = ops.exp(ops.arange(half_dim, dtype=mindspore.int64).float() * -emb)
        emb = ops.arange(num_embeddings, dtype=mindspore.int64).\
                  float().unsqueeze(1) * emb.unsqueeze(0)
        emb = ops.cat([ops.sin(emb), ops.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = ops.cat([emb, ops.zeros((num_embeddings, 1))], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0

        return emb.to(mindspore.float32)

    def forward(self, input_ids: mindspore.Tensor, past_key_values_length: int = 0):
        bsz, seq_len = input_ids.shape
        # Create the position ids from the input token ids. Any padded tokens remain padded.
        position_ids = self.create_position_ids_from_input_ids(
            input_ids, self.padding_idx, past_key_values_length)

        # expand embeddings if needed
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.shape[0]:
            # recompute/expand embeddings if needed
            self.weights = self.get_embedding(max_pos, self.embedding_dim, self.padding_idx)

        x = self.weights.index_select(0, position_ids.view(-1)).view(bsz, seq_len, -1)

        return x

    def create_position_ids_from_input_ids(
            self, input_ids: mindspore.Tensor, padding_idx: int,
            past_key_values_length: Optional[int] = 0
    ):
        """
        Replace non-padding symbols with their position numbers.
         Position numbers begin at padding_idx+1. Padding
        symbols are ignored. This is modified from
        fairseq's `utils.make_positions`.
        """
        # The series of casts and type-conversions here
        # are carefully balanced to both work with ONNX export and XLA.
        mask = input_ids.ne(padding_idx).int()
        incremental_indices = (ops.cumsum(mask, dim=1).
                               type_as(mask) + past_key_values_length) * mask
        return incremental_indices.long() + padding_idx


class TrOCRAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper."""

    def __init__(
            self,
            config,
            embed_dim: int,
            num_heads: int,
            kdim: int = None,
            vdim: int = None,
            dropout: float = 0.0,
            is_decoder: bool = False,
            bias: bool = True,
            is_cross_attention: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        if not self.head_dim * num_heads == self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads"
                f" (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {num_heads})."
            )
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: mindspore.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).swapaxes(1, 2)

    def forward(
            self,
            hidden_states: mindspore.Tensor,
            key_value_states: Optional[mindspore.Tensor] = None,
            past_key_value: Optional[Tuple[mindspore.Tensor]] = None,
            attention_mask: Optional[mindspore.Tensor] = None,
            layer_head_mask: Optional[mindspore.Tensor] = None,
            output_attentions: bool = False,
    ) -> Tuple[mindspore.Tensor, Optional[mindspore.Tensor], Optional[Tuple[mindspore.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, embed_dim = hidden_states.shape

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = ops.cat([past_key_value[0], key_states], dim=2)
            value_states = ops.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(mindspore.Tensor,
            # mindspore.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer
            # can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder)
            # save Tuple(mindspore.Tensor, mindspore.Tensor) of
            # all previous decoder key/value_states.
            # Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states
            # to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention
            # `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.shape[1]
        attn_weights = ops.bmm(query_states, key_states.swapaxes(1, 2))

        if attn_weights.shape != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size "
                f"{(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.shape}"
            )

        if attention_mask is not None:
            if attention_mask.shape != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size "
                    f"{(bsz, 1, tgt_len, src_len)}, but is {attention_mask.shape}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = ops.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.shape != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.shape}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * \
                           attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = ops.bmm(attn_probs, value_states)

        if attn_output.shape != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size "
                f"{(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.shape}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.swapaxes(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class TrOCRDecoderLayer(nn.Module):
    """
    A class of TrOCR decoder layer.
    """
    def __init__(self, config: TrOCRConfig):
        super().__init__()
        self.embed_dim = config.hidden_size

        self.self_attn = TrOCRAttention(
            config,
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm([self.embed_dim])

        if config.is_decoder:
            self.encoder_attn = TrOCRAttention(
                config,
                embed_dim=self.embed_dim,
                num_heads=config.decoder_attention_heads,
                kdim=config.cross_attention_hidden_size,
                vdim=config.cross_attention_hidden_size,
                dropout=config.attention_dropout,
                is_decoder=True,
                is_cross_attention=True,
            )
            self.encoder_attn_layer_norm = nn.LayerNorm([self.embed_dim])

        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm([self.embed_dim])

    def forward(
            self,
            hidden_states: mindspore.Tensor,
            attention_mask: Optional[mindspore.Tensor] = None,
            encoder_hidden_states: Optional[mindspore.Tensor] = None,
            encoder_attention_mask: Optional[mindspore.Tensor] = None,
            layer_head_mask: Optional[mindspore.Tensor] = None,
            cross_attn_layer_head_mask: Optional[mindspore.Tensor] = None,
            past_key_value: Optional[Tuple[mindspore.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = True,
    ):
        """
        Args:
            hidden_states (`mindspore.Tensor`):
                input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`mindspore.Tensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where
                padding elements are indicated by very large negative values.
            encoder_hidden_states (`mindspore.Tensor`):
                cross attention input to the layer of
                shape `(batch, seq_len, embed_dim)`
            encoder_attention_mask (`mindspore.Tensor`): encoder attention
                mask of size
                `(batch, 1, tgt_len, src_len)` where padding
                 elements are indicated by very large negative values.
            layer_head_mask (`mindspore.Tensor`): mask for
                attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            cross_attn_layer_head_mask (`mindspore.Tensor`): mask for
                cross-attention heads in a given layer of
                size *(decoder_attention_heads,)*.
            past_key_value (`Tuple(mindspore.Tensor)`):
                cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of
                all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )

        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None

        if encoder_hidden_states is not None:
            residual = hidden_states

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )

            hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states,
                                    p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)

        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class TrOCRPreTrainedModel(PreTrainedModel):
    config_class = TrOCRConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["TrOCRDecoderLayer"]

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.assign_value(initializer(
                Normal(sigma=std, mean=0.0), module.weight.shape, module.weight.dtype))
            if module.bias is not None:
                module.bias.assign_value(initializer('zeros', module.bias.shape, module.bias.dtype))
        elif isinstance(module, nn.Embedding):
            emb_weight = np.random.normal(0, std, module.weight.shape)
            if module.padding_idx is not None:
                emb_weight[module.padding_idx] = 0
            module.weight.assign_value(mindspore.Tensor(emb_weight, module.weight.dtype))


class TrOCRDecoder(TrOCRPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers*
    layers. Each layer is a [`TrOCRDecoderLayer`]

    Args:
        config: TrOCRConfig
    """

    def __init__(self, config: TrOCRConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        embed_scale = math.sqrt(config.hidden_size) if config.scale_embedding else 1.0

        self.embed_tokens = TrOCRScaledWordEmbedding(
            config.vocab_size, config.hidden_size, self.padding_idx, embed_scale=embed_scale
        )

        if config.use_learned_position_embeddings:
            self.embed_positions = TrOCRLearnedPositionalEmbedding(
                config.max_position_embeddings, config.hidden_size)
        else:
            self.embed_positions = TrOCRSinusoidalPositionalEmbedding(
                config.max_position_embeddings + self.padding_idx + 1,
                config.hidden_size,
                self.padding_idx,
            )

        if config.layernorm_embedding:
            self.layernorm_embedding = nn.LayerNorm([config.hidden_size])
        else:
            self.layernorm_embedding = None

        self.layers = nn.ModuleList([TrOCRDecoderLayer(config) for _ in range(config.decoder_layers)])

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            head_mask=None,
            cross_attn_head_mask=None,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        Args:
            input_ids (`mindspore.Tensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
                 Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`].
                See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`mindspore.Tensor` of shape
            `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token
                indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            encoder_hidden_states
            (`mindspore.Tensor` of shape `(batch_size,
            encoder_sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last
                 layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (`mindspore.Tensor` of
            shape `(batch_size, encoder_sequence_length)`, *optional*):
                Mask to avoid performing cross-attention on padding
                 tokens indices of encoder input_ids. Mask values
                selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`mindspore.Tensor` of shape
            `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention
                modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            cross_attn_head_mask (`mindspore.Tensor` of shape
            `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention
                modules in encoder to avoid performing cross-attention
                on hidden heads. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(mindspore.Tensor))`, *optional*,
            returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(mindspore.Tensor)` of length
                `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length,
                embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length,
                 embed_size_per_head)`.

                Contains pre-computed hidden-states (key and values in
                 the self-attention blocks and in the
                cross-attention blocks) that can be used (see
                `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally
                 input only the last `decoder_input_ids` (those
                that don't have their past key value states given to
                this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            inputs_embeds (`mindspore.Tensor` of shape
            `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can
                choose to directly pass an embedded representation.
                This is useful if you want more control over how to
                convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of
                all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all
                layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`]
                instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions\
                                                 is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids"
                             " and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input = input_ids
            input_ids = input_ids.view(-1, input.shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
            input = inputs_embeds[:, :, -1]
        else:
            raise ValueError("You have to specify "
                             "either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2]\
            if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self.config.use_learned_position_embeddings:
            embed_pos = self.embed_positions(input, past_key_values_length=past_key_values_length)
        else:
            embed_pos = self.embed_positions(input_ids,
                                             past_key_values_length=past_key_values_length)

        hidden_states = inputs_embeds + embed_pos

        if self.layernorm_embedding is not None:
            hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)

        input_shape = input.shape

        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _prepare_4d_attention_mask(
                encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning(
                    "`use_cache = True` is incompatible with "
                    "gradient checkpointing. Setting `use_cache = False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states
                                      is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct
        # number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask],
                                        ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                if attn_mask.shape[0] != (len(self.layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} "
                        f"layers, but it is for"
                        f" {head_mask.shape[0]}."
                    )
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.training:
                dropout_probability = ops.rand(1)
                if dropout_probability < self.layerdrop:
                    continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache,
                          all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


class TrOCRDecoderWrapper(TrOCRPreTrainedModel):
    """
    The TrOCR Model with a language modeling head.
    Can be used for summarization.

    This model inherits from [`PreTrainedModel`].
    Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or
    saving, resizing the input embeddings, pruning heads
    etc.)

    Parameters:
        config ([`TrOCRConfig`]):
            Model configuration class with all the parameters of the model.
             Initializing with a config file does not
            load the weights associated with the model,
            only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method
            to load the model weights.

    This wrapper class is a helper class to correctly load pretrained
    checkpoints when the causal language model is
    used in combination with the [`EncoderDecoderModel`] framework.
    """

    def __init__(self, config):
        super().__init__(config)
        self.decoder = TrOCRDecoder(config)

    def forward(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)


class TrOCRForCausalLM(TrOCRPreTrainedModel):
    """
    The TrOCR Decoder with a language modeling head.
    Can be used as the decoder part of [`EncoderDecoderModel`] and
    [`VisionEncoderDecoder`].

    This model inherits from [`PreTrainedModel`].
    Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving,
     resizing the input embeddings, pruning heads
    etc.)

    Parameters:
        config ([`TrOCRConfig`]):
            Model configuration class with all the parameters of the model.
            Initializing with a config file does not
            load the weights associated with the model,
            only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method
            to load the model weights.
    """
    _tied_weights_keys = ["output_projection.weight"]

    def __init__(self, config):
        config = copy.deepcopy(config)
        config.is_decoder = True
        config.is_encoder_decoder = False
        super().__init__(config)
        self.model = TrOCRDecoderWrapper(config)

        self.output_projection = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value

    def get_output_embeddings(self):
        return self.output_projection

    def set_output_embeddings(self, new_embeddings):
        self.output_projection = new_embeddings

    def set_decoder(self, decoder):
        self.model.decoder = decoder

    def get_decoder(self):
        return self.model.decoder

    def forward(
            self,
            input_ids: Optional[mindspore.Tensor] = None,
            attention_mask: Optional[mindspore.Tensor] = None,
            encoder_hidden_states: Optional[mindspore.Tensor] = None,
            encoder_attention_mask: Optional[mindspore.Tensor] = None,
            head_mask: Optional[mindspore.Tensor] = None,
            cross_attn_head_mask: Optional[mindspore.Tensor] = None,
            past_key_values: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
            inputs_embeds: Optional[mindspore.Tensor] = None,
            labels: Optional[mindspore.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        Args:
            input_ids (`mindspore.Tensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be
                ignored by default should you provide it.

                Indices can be obtained using [`AutoTokenizer`].
                See [`PreTrainedTokenizer.encode`]
                and [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`mindspore.Tensor` of shape
            `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices.
                Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            encoder_hidden_states
            (`mindspore.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder.
                Used in the cross-attention if the model is configured as a decoder.
            encoder_attention_mask (`mindspore.Tensor` of
            shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on the padding token
                indices of the encoder input. This mask is used
                in the cross-attention if the model is configured
                 as a decoder. Mask values selected in `[0, 1]`:
            head_mask (`mindspore.Tensor` of shape `(decoder_layers,
            decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules.
                 Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            cross_attn_head_mask (`mindspore.Tensor` of shape
            `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the cross-attention modules.
                Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(mindspore.Tensor))`, *optional*,
            returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(mindspore.Tensor)` of length `config.n_layers`,
                with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`)
                and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
                 The two additional
                tensors are only required when the model is used as a
                decoder in a Sequence to Sequence model.

                Contains pre-computed hidden-states (key and values in
                the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input)
                to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input
                 only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model)
                 of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            labels (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss.
                Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring).
                 Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels
                in `[0, ..., config.vocab_size]`.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and
                can be used to speed up decoding (see `past_key_values`).

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
                See `attentions` under returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states`
                under returned tensors for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        Returns:
            mindnlp.transformers.modeling_outputs.CausalLMOutputWithCrossAttentions
        """
        output_attentions = output_attentions if output_attentions is not None \
            else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = self.output_projection(outputs[0])

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None,
            attention_mask=None, use_cache=None, **kwargs
    ):
        # if model is used as a decoder in encoder-decoder model,
        # the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)

        if past_key_values:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]
        # first step, decoder_cached_states are empty
        return {
            "input_ids": input_ids,  # encoder_outputs is defined. input_ids not needed
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),
            )
        return reordered_past


__all__ = [
    'TrOCRDecoder',
    'TrOCRPreTrainedModel',
    'TrOCRForCausalLM',
]
