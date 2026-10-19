"""Explicit runtime precision policy for a separately reported carrier rerun.

Q/K/V attention computation is FP32; the rest of the frozen model still uses
its original AMP policy. No seed, optimizer, architecture or loss is changed.
"""
from functools import wraps
import torch


def install_attention_precision_guard(model):
    count = 0
    for module in model.modules():
        if not isinstance(module, torch.nn.MultiheadAttention):
            continue
        if getattr(module, '_reproduction_fp32_attention', False):
            continue
        original = module.forward

        def make_forward(function):
            @wraps(function)
            def forward(query, key, value, **kwargs):
                # The frozen carrier calls MHA with positional Q/K/V and
                # keyword options. Boolean masks retain their semantics.
                for field in ('key_padding_mask', 'attn_mask'):
                    mask = kwargs.get(field)
                    if isinstance(mask, torch.Tensor) and mask.is_floating_point():
                        kwargs[field] = mask.float()
                with torch.autocast(query.device.type, enabled=False):
                    return function(query.float(), key.float(), value.float(), **kwargs)
            return forward

        module.forward = make_forward(original)
        module._reproduction_fp32_attention = True
        count += 1
    return count
