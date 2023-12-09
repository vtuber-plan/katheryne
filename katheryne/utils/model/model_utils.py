import os
import math
import torch
from transformers import (
    AutoConfig,
    AutoModel,
    PreTrainedModel,
)
from huggingface_hub import snapshot_download
from transformers.deepspeed import HfDeepSpeedConfig

def create_hf_model(model_class,
                    model_name_or_path,
                    tokenizer,
                    dtype=None,
                    disable_dropout=False,
                    trust_remote_code=True) -> PreTrainedModel:
    model_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    if disable_dropout:
        model_config.dropout = 0.0

    model = model_class.from_pretrained(
        model_name_or_path,
        from_tf=bool(".ckpt" in model_name_or_path),
        config=model_config,
        trust_remote_code=trust_remote_code,
        torch_dtype=dtype,
    )

    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(int(8 * math.ceil(len(tokenizer) / 8.0)))  # make the vocab size multiple of 8

    return model

def save_hf_format(model, tokenizer=None, output_dir: str="./", sub_folder: str=""):
    # used to save huggingface format, so we can use it for hf.from_pretrained
    model_to_save = model.module if hasattr(model, 'module') else model
    CONFIG_NAME = "config.json"
    WEIGHTS_NAME = "pytorch_model.bin"
    output_dir = os.path.join(output_dir, sub_folder)
    os.makedirs(output_dir, exist_ok=True)
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    save_dict = model_to_save.state_dict()
    for key in list(save_dict.keys()):
        if "lora" in key:
            del save_dict[key]
    torch.save(save_dict, output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    if tokenizer is not None:
        try:
            tokenizer.save_vocabulary(output_dir)
        except NotImplementedError:
            pass
