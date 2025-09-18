# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
from models.discrete_unet import DiscreteUNetModel, DiscretePrimeUNetModel
from models.ema import EMA

MODEL_CONFIGS = {
    "model_discrete": {
        "in_channels": 3,
        "model_channels": 96,
        "out_channels": 3,
        "num_res_blocks": 5,
        "attention_resolutions": [2],
        "dropout": 0.4,
        "channel_mult": [3, 4, 4],
        "conv_resample": False,
        "dims": 2,
        "num_classes": None,
        "use_checkpoint": False,
        "num_heads": -1,
        "num_head_channels": 64,
        "num_heads_upsample": -1,
        "use_scale_shift_norm": True,
        "resblock_updown": False,
        "use_new_attention_order": True,
        "with_fourier_features": False,
    },
    "model_discrete_supertoken": {
        "in_channels": 2,
        "model_channels": 96,
        "out_channels": 2,
        "num_res_blocks": 5,
        "attention_resolutions": [2],
        "dropout": 0.4,
        "channel_mult": [3, 4, 4],
        "conv_resample": False,
        "dims": 2,
        "num_classes": None,
        "use_checkpoint": False,
        "num_heads": -1,
        "num_head_channels": 64,
        "num_heads_upsample": -1,
        "use_scale_shift_norm": True,
        "resblock_updown": False,
        "use_new_attention_order": True,
        "with_fourier_features": False,
    },
}

def instantiate_model(use_ema: bool, super_token: bool = False):
    config = MODEL_CONFIGS["model_discrete" + ("_supertoken" if super_token else "")]
    if super_token:
        model = DiscreteUNetModel(
            vocab_size=4097,
            **config,
        )
    else:
        model = DiscreteUNetModel(
            vocab_size=257,
            **config,
        )

    if use_ema:
        return EMA(model=model)
    else:
        return model

def instantiate_prime_model(use_ema: bool, target_length: int, base: int, vocab_size: int = 256):
    config = MODEL_CONFIGS["model_discrete"]
    model = DiscretePrimeUNetModel(
        vocab_size=vocab_size,
        target_length=target_length,
        base=base+1,
        **config,
    )
    if use_ema:
        return EMA(model=model)
    else:
        return model