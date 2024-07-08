import json
import os
import math
from typing import Any, Dict, Literal, Optional
import torch
from transformers import (
    AutoConfig,
    AutoModel,
    PreTrainedModel,
)
from huggingface_hub import snapshot_download
from peft import PeftModel, PeftModelForCausalLM

from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification

def create_hf_model(model_class,
                    model_name_or_path,
                    dtype=None,
                    disable_dropout=False,
                    trust_remote_code=True,
                    atten_class: Literal["eager", "flash", "sdpa"]=False,
                    model_kwargs: Optional[Dict[str, Any]]=None) -> PreTrainedModel:
    more_args = {}
    if atten_class == "eager":
        more_args["attn_implementation"] = "eager"
    elif atten_class == "flash":
        more_args["attn_implementation"] = "flash_attention_2"
    elif atten_class == "sdpa":
        more_args["attn_implementation"] = "sdpa"
    else:
        raise Exception("Unknown attention class")
    
    model_json = os.path.join(model_name_or_path, "config.json")
    adapter_model_json = os.path.join(model_name_or_path, "adapter_config.json")

    if os.path.exists(adapter_model_json):
        model_json_file = json.load(open(adapter_model_json))
        base_model_name = model_json_file["base_model_name_or_path"]
        model_config = AutoConfig.from_pretrained(base_model_name, trust_remote_code=True)
    else:
        model_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)

    if disable_dropout:
        model_config.dropout = 0.0
    
    if model_kwargs is None:
        model_kwargs_dict = {}
    else:
        model_kwargs_dict = model_kwargs
    
    if model_class in [AutoModelForTokenClassification]:
        if "num_labels" not in model_kwargs_dict:
            model_config.num_labels = 1
            print("Set num_labels of model as 1.")

    if model_class in [AutoModelForSequenceClassification]:
        if "num_labels" not in model_kwargs_dict:
            model_config.num_labels = 1
            print("Set num_labels of model as 1.")
    model = model_class.from_pretrained(
        model_name_or_path,
        from_tf=bool(".ckpt" in model_name_or_path),
        config=model_config,
        trust_remote_code=trust_remote_code,
        torch_dtype=dtype,
        **model_kwargs_dict,
    )
    if os.path.exists(adapter_model_json):
        from peft import (
            PeftModelForCausalLM,
            PeftModelForSeq2SeqLM,
            PeftModelForSequenceClassification,
            PeftModelForTokenClassification,
            PeftModelForQuestionAnswering,
            PeftModelForFeatureExtraction,
        )
        if "CausalLM" in model.__class__.__name__:
            peft_model = PeftModelForCausalLM.from_pretrained(model, model_name_or_path, is_trainable=True)
        elif "Seq2Seq" in model.__class__.__name__:
            peft_model = PeftModelForSeq2SeqLM.from_pretrained(model, model_name_or_path, is_trainable=True)
        elif "SequenceClassification" in model.__class__.__name__:
            peft_model = PeftModelForSequenceClassification.from_pretrained(model, model_name_or_path, is_trainable=True)
        elif "TokenClassification" in model.__class__.__name__:
            peft_model = PeftModelForTokenClassification.from_pretrained(model, model_name_or_path, is_trainable=True)
        elif "QuestionAnswering" in model.__class__.__name__:
            peft_model = PeftModelForQuestionAnswering.from_pretrained(model, model_name_or_path, is_trainable=True)
        elif "FeatureExtraction" in model.__class__.__name__:
            peft_model = PeftModelForFeatureExtraction.from_pretrained(model, model_name_or_path, is_trainable=True)
        else:
            raise Exception("Unknown Type of Pretrained Model. Failed to load adaptor.")
        if isinstance(peft_model, PeftModel) or isinstance(peft_model, PeftModelForCausalLM):
            model = peft_model.merge_and_unload()
    # model.config.end_token_id = tokenizer.eos_token_id
    # model.config.pad_token_id = model.config.eos_token_id
    # model.resize_token_embeddings(int(8 * math.ceil(len(tokenizer) / 8.0)))  # make the vocab size multiple of 8

    return model

def save_hf_format(model, tokenizer=None, output_dir: str="./", sub_folder: str="", peft_merge=False):
    output_dir = os.path.join(output_dir, sub_folder)
    os.makedirs(output_dir, exist_ok=True)
    
    if isinstance(model, PeftModel) or isinstance(model, PeftModelForCausalLM):
        if peft_merge:
            merged_model = model.merge_and_unload()
        else:
            merged_model = model
        merged_model.save_pretrained(output_dir)
    elif isinstance(model, PreTrainedModel):
        model.save_pretrained(output_dir)
    else:
        raise Exception("Unsupported model to save.")


def save_hf_format_native(model, tokenizer=None, output_dir: str="./", sub_folder: str=""):
    output_dir = os.path.join(output_dir, sub_folder)
    os.makedirs(output_dir, exist_ok=True)

    # used to save huggingface format, so we can use it for hf.from_pretrained
    model_to_save = model.module if hasattr(model, 'module') else model
    CONFIG_NAME = "config.json"
    WEIGHTS_NAME = "pytorch_model.bin"
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    save_dict = model_to_save.state_dict()
    torch.save(save_dict, output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    if tokenizer is not None:
        try:
            tokenizer.save_vocabulary(output_dir)
        except NotImplementedError:
            pass
