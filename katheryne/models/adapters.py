# coding=utf-8
# Copyright 2024 XiaHan
# 
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from typing import List, Optional
from transformers import PreTrainedModel
from peft import LoftQConfig, LoraConfig, get_peft_model, PeftModel


__all__ = [
    "setup_lora",
]

def setup_lora(
        base_model: PreTrainedModel,
        r: int=128,
        target_modules: Optional[List[str]]=None,
        lora_alpha: int=8,
        lora_dropout: float=0.0,
        fan_in_fan_out: bool=False,
        bias: str="none",
        loftq_config: dict=None,
        use_dora: bool=False,
        task_type: str="CAUSAL_LM"
    ) -> PeftModel:
    # set 4bit quantization
    if loftq_config is not None:
        loftq_config = LoftQConfig(
            loftq_bits=loftq_config.get("loftq_bits", 4),
            loftq_iter=loftq_config.get("loftq_iter", 1)
        )
    else:
        loftq_config = {}
    lora_config = LoraConfig(
        task_type=task_type,
        r=r,
        target_modules=target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        fan_in_fan_out=fan_in_fan_out,
        bias=bias,
        loftq_config=loftq_config,
        use_dora=use_dora,
    )
    model = get_peft_model(base_model, lora_config)
    return model
