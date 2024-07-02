# coding=utf-8
# Copyright 2024 XiaHan
# 
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import argparse
import os
import warnings

import lightning_fabric
import torch

import tqdm
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModel,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from katheryne.models.adapters import setup_lora
from katheryne.utils.hparams import HParams
from katheryne.utils.model.model_utils import create_hf_model
from katheryne.utils.model.tokenizer_utils import load_hf_tokenizer
from katheryne.utils.utils import parse_dtype_str

def rlhf_train(args: argparse.Namespace, hparams: HParams, create_dataset):
    torch.autograd.set_detect_anomaly(True)
    master_port = os.environ.get("MASTER_PORT", None)
    master_addr = os.environ.get("MASTER_ADDR", None)
    world_size = os.environ.get("WORLD_SIZE", None)
    rank = os.environ.get("RANK", None)

    # If passed along, set the training seed now.
    lightning_fabric.seed_everything(args.seed)

    # Load tokenizer
    tokenizer = load_hf_tokenizer(hparams.model_name_or_path, fast_tokenizer=True)
    tokenizer_path = hparams.get("tokenizer_path", hparams.model_name_or_path)
    
    # Load model
    model_path = None
    if "model_name_or_path" in hparams:
        model_path = hparams.model_name_or_path
    elif "model_name" in hparams:
        model_path = hparams.model_name
    
    # Get torch type
    torch_dtype_str = hparams.get("model_torch_dtype", "auto")
    if torch_dtype_str != "auto":
        torch_dtype = parse_dtype_str(torch_dtype_str)
    else:
        torch_dtype = torch_dtype_str
    
    if torch_dtype == torch.bfloat16 and args.accelerator in ["cpu"]:
        raise RuntimeError("Models in bfloat16 cannot run with the accelerator CPU.")
    if torch_dtype == torch.float16 and args.accelerator in ["cpu"]:
        raise RuntimeError("Models in float16 cannot run with the accelerator CPU.")
    
     # Create Model
    model = create_hf_model(
        model_class=AutoModelForCausalLMWithValueHead,
        model_name_or_path=model_path,
        dtype=torch_dtype,
        disable_dropout=hparams.disable_dropout,
        atten_class=hparams.get("atten_class", "eager"),
        model_kwargs=hparams.get("model_kwargs", {}),
    )

    # Setup LORA
    if "lora" in hparams:
        model = setup_lora(
            model,
            r=hparams.lora.get("r", 128),
            target_modules=hparams.lora.get("target_modules", []),
            lora_alpha=hparams.lora.get("lora_alpha", 8),
            lora_dropout=hparams.lora.get("lora_dropout", 0.0),
            fan_in_fan_out=hparams.lora.get("fan_in_fan_out", False),
            bias=hparams.lora.get("bias", 'none'),
            loftq_config=hparams.lora.get("loftq", None),
            use_dora=hparams.lora.get("use_dora", False),
            task_type=hparams.lora.get("task_type", "CAUSAL_LM")
        )
        if hparams.get("gradient_checkpointing", False):
            model.enable_input_require_grads()

    # Create ref model
    ref_model = create_hf_model(
        model_class=AutoModelForCausalLMWithValueHead,
        model_name_or_path=model_path,
        dtype=torch_dtype,
        disable_dropout=hparams.disable_dropout,
        atten_class=hparams.get("atten_class", "eager"),
        model_kwargs=hparams.get("model_kwargs", {}),
    )

    # Load Reward Model
    reward_model = create_hf_model(
        model_class=AutoModelForCausalLMWithValueHead,
        model_name_or_path=hparams.get("reward_model_path", None),
        dtype=torch_dtype,
        disable_dropout=hparams.disable_dropout,
        atten_class=hparams.get("atten_class", "eager"),
        model_kwargs=hparams.get("model_kwargs", {}),
    )

    # Create Config
    config_params = {}
    config_params["exp_name"] = hparams.get("exp_name", "trl_train")
    config_params["seed"] = args.seed

    loggers = hparams.get("logger", [{"logger_type": "tb", "save_dir": "lightning_logs"}])
    if isinstance(loggers, list) or isinstance(loggers, tuple):
        loggers = loggers
        if len(loggers) > 1:
            raise Exception("Unsupported multi loggers in trl")
    elif isinstance(loggers, dict):
        loggers = [loggers]
    else:
        raise Exception("Unsupported type in logger field")
    for logger in loggers:
        logger_type = logger.get("logger_type", "tb")
        save_dir = logger.get("save_dir", "logs")
        logger_save_dir = os.path.join(args.path, save_dir)

        if logger_type.lower() in ["tb", "tensorboard"]:
            config_params["log_with"] = "tensorboard"
        elif logger_type.lower() in ["wandb"]:
            config_params["log_with"] = "wandb"
        else:
            raise Exception("Unsupported logger type.")
        break

    config_params["task_name"] = hparams.get("task_name", None)
    config_params["model_name"] = hparams.get("model_name_or_path", None)
    config_params["query_dataset"] = hparams.get("data_path", None)

    config_params["learning_rate"] = hparams.get("learning_rate", 1.41e-5)
    config_params["gradient_checkpointing"] = hparams.get("gradient_checkpointing", False)

    config = PPOConfig(**config_params)

    ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator)

    output_length_sampler = LengthSampler(hparams.get("output_min_length", 16), hparams.get("output_max_length", 1024))

    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
    }

    for epoch, batch in tqdm.tqdm(enumerate(ppo_trainer.dataloader)):
        query_tensors = batch["input_ids"]

        response_tensors = []
        for query in query_tensors:
            gen_len = output_length_sampler()
            generation_kwargs["max_new_tokens"] = gen_len
            response = ppo_trainer.generate(query, **generation_kwargs)
            response_tensors.append(response.squeeze()[-gen_len:])
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

        # Compute reward score
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        rewards = reward_model.forward(texts)

        # Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)





