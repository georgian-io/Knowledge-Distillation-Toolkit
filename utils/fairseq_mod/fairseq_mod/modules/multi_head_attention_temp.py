"""

We simplified the logics in torch.nn.functional.multi_head_attention_forward, and only kept operations that are relevant to wav2vec 2.0.

"""

from __future__ import division

import warnings
import math

import torch
from torch._C import _infer_size, _add_docstr
from torch.nn import _reduction as _Reduction
from torch.nn.modules import utils
from torch.nn.modules.utils import _single, _pair, _triple, _list_with_default
from torch.nn import grad  # noqa: F401
from torch import _VF
from torch._jit_internal import boolean_dispatch, List, Optional, _overload
from torch._overrides import has_torch_function, handle_torch_function
from torch.nn.functional import linear, softmax, dropout

Tensor = torch.Tensor

def multi_head_attention_forward_temp(query: Tensor,
                                 key: Tensor,
                                 value: Tensor,
                                 embed_dim_to_check: int,
                                 num_heads: int,
                                 in_proj_weight: Tensor,
                                 in_proj_bias: Tensor,
                                 bias_k: Optional[Tensor],
                                 bias_v: Optional[Tensor],
                                 add_zero_attn: bool,
                                 dropout_p: float,
                                 out_proj_weight: Tensor,
                                 out_proj_bias: Tensor,
                                 training: bool = True,
                                 key_padding_mask: Optional[Tensor] = None,
                                 need_weights: bool = True,
                                 attn_mask: Optional[Tensor] = None,
                                 use_separate_proj_weight: bool = False,
                                 q_proj_weight: Optional[Tensor] = None,
                                 k_proj_weight: Optional[Tensor] = None,
                                 v_proj_weight: Optional[Tensor] = None,
                                 static_k: Optional[Tensor] = None,
                                 static_v: Optional[Tensor] = None,
                                 q_proj_bias = None,
                                 k_proj_bias = None,
                                 v_proj_bias = None
                                 ):
    if not torch.jit.is_scripting():
        tens_ops = (query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v,
                    out_proj_weight, out_proj_bias)
        if any([type(t) is not Tensor for t in tens_ops]) and has_torch_function(tens_ops):
            return handle_torch_function(
                multi_head_attention_forward, tens_ops, query, key, value,
                embed_dim_to_check, num_heads, in_proj_weight, in_proj_bias,
                bias_k, bias_v, add_zero_attn, dropout_p, out_proj_weight,
                out_proj_bias, training=training, key_padding_mask=key_padding_mask,
                need_weights=need_weights, attn_mask=attn_mask,
                use_separate_proj_weight=use_separate_proj_weight,
                q_proj_weight=q_proj_weight, k_proj_weight=k_proj_weight,
                v_proj_weight=v_proj_weight, static_k=static_k, static_v=static_v)
    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    # allow MHA to have different sizes for the feature dimension
    assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5

    # !!! Assume use_separate_proj_weight = True
    q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
    len1, len2 = q_proj_weight_non_opt.size()
    assert len1 == embed_dim and len2 == query.size(-1)

    k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
    len1, len2 = k_proj_weight_non_opt.size()
    assert len1 == embed_dim and len2 == key.size(-1)

    v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
    len1, len2 = v_proj_weight_non_opt.size()
    assert len1 == embed_dim and len2 == value.size(-1)

    # !!! Did not indexing in_proj_bias because biases are supplied separately
    q = linear(query, q_proj_weight_non_opt, q_proj_bias)
    k = linear(key, k_proj_weight_non_opt, k_proj_bias)
    v = linear(value, v_proj_weight_non_opt, v_proj_bias)

    q = q * scaling

   # !!! Assumed that attn_mask is None
   # !!! Assumed that key_padding_mask.dtype = torch.Bool
   # !!! Assumed that bias_k and bias_v are None

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    if static_k is not None:
        assert static_k.size(0) == bsz * num_heads
        assert static_k.size(2) == head_dim
        k = static_k

    if static_v is not None:
        assert static_v.size(0) == bsz * num_heads
        assert static_v.size(2) == head_dim
        v = static_v

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len

    # !!! Assumed add_zero_attn is False

    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float('-inf'),
        )
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

    attn_output_weights = softmax(
        attn_output_weights, dim=-1)
    attn_output_weights = dropout(attn_output_weights, p=dropout_p, training=training)

    attn_output = torch.bmm(attn_output_weights, v)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None

