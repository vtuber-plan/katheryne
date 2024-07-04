# coding=utf-8
# Copyright 2024 XiaHan
# 
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import argparse
import math
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

from katheryne.data.collators import DataCollatorWithPadding
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

    # Model and Tokenizer Path
    model_path = None
    if "model_name_or_path" in hparams:
        model_path = hparams.model_name_or_path
    elif "model_path" in hparams:
        model_path = hparams.model_path
    else:
        raise Exception("The model path or name is not found in the hparams file.")

    if "tokenizer_path" in hparams:
        tokenizer_path = hparams.get("tokenizer_path", model_path)
    else:
        tokenizer_path = model_path

    # Load tokenizer
    tokenizer = load_hf_tokenizer(tokenizer_path, fast_tokenizer=True, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

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
    model_class_config = hparams.get("model_class", "AutoModelForCausalLM")
    if model_class_config == "AutoModelForCausalLM":
        model_class = AutoModelForCausalLM
    elif model_class_config == "AutoModelForSequenceClassification":
        model_class = AutoModelForSequenceClassification
    elif model_class_config == "AutoModelForTokenClassification":
        model_class = AutoModelForTokenClassification
    elif model_class_config == "AutoModel":
        model_class = AutoModel
    else:
        raise Exception("Unsupported model class config.")
    model = create_hf_model(
        model_class=model_class_config,
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
        model_class=model_class_config,
        model_name_or_path=model_path,
        dtype=torch_dtype,
        disable_dropout=hparams.disable_dropout,
        atten_class=hparams.get("atten_class", "eager"),
        model_kwargs=hparams.get("model_kwargs", {}),
    )

    # Reward Model and Tokenizer Path
    reward_model_path = hparams.get("reward_model_path", None)
    if "tokenizer_path" in hparams:
        reward_tokenizer_path = hparams.get("reward_tokenizer_path", reward_model_path)
    else:
        reward_tokenizer_path = reward_model
    
    # Load Reward Tokenizer
    reward_tokenizer = load_hf_tokenizer(reward_tokenizer_path, fast_tokenizer=True, padding_side="left")
    reward_tokenizer.pad_token = reward_tokenizer.eos_token
    reward_tokenizer.pad_token_id = reward_tokenizer.eos_token_id

    # Load Reward Model
    reward_model = create_hf_model(
        model_class=AutoModelForCausalLMWithValueHead,
        model_name_or_path=hparams.get("reward_model_path", None),
        dtype=torch_dtype,
        disable_dropout=hparams.disable_dropout,
        atten_class=hparams.get("atten_class", "eager"),
        model_kwargs=hparams.get("model_kwargs", {}),
    )

    # Prepare the data
    print("***** Prepare Dataset *****")
    train_dataset, valid_dataset = create_dataset(
        hparams=hparams,
        data_path=hparams.data_path,
        tokenizer_path=tokenizer_path,
        max_seq_len=hparams.max_seq_len,
    )

    # DataLoaders creation:
    print("***** DataLoaders creation *****")
    collator = DataCollatorWithPadding(
        tokenizer=load_hf_tokenizer(tokenizer_path, fast_tokenizer=True, show_info=True, padding_side="left"),
        padding="longest",
        max_length=hparams.max_seq_len
    )
 
    # Create Config
    config_params = {}
    config_params["tracker_kwargs"] = {}
    config_params["accelerator_kwargs"] = {}
    config_params["project_kwargs"] = {}

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
        elif logger_type.lower() in ["comet", "comet_ml"]:
            config_params["log_with"] = "comet_ml"
        else:
            raise Exception("Unsupported logger type.")
        config_params["project_kwargs"]["logging_dir"] = logger_save_dir
        break
    
    config_params["accelerator_kwargs"]["device_placement"] = True
    if "fp16" in hparams and hparams.fp16:
        print("using fp16")
        precision = "fp16"
        assert (args.accelerator not in ["cpu"]), "models in float16 cannot run with the accelerator CPU."
    elif "bf16" in hparams and hparams.bf16:
        print("using bf16")
        precision = "bf16"
        assert (args.accelerator not in ["cpu"]), "models in bfloat16 cannot run with the accelerator CPU."
    else:
        print("using fp32")
        precision = "no"
    config_params["accelerator_kwargs"]["mixed_precision"] = precision

    if args.accelerator == "cpu":
        config_params["accelerator_kwargs"]["cpu"] = True
    else:
        config_params["accelerator_kwargs"]["cpu"] = False
    config_params["accelerator_kwargs"]["mixed_precision"] = precision
    config_params["accelerator_kwargs"]["mixed_precision"] = precision

    config_params["task_name"] = hparams.get("task_name", None)
    config_params["model_name"] = hparams.get("model_name_or_path", None)
    # config_params["query_dataset"] = hparams.get("data_path", None)

    config_params["learning_rate"] = hparams.get("learning_rate", 1.41e-5)
    config_params["gradient_checkpointing"] = hparams.get("gradient_checkpointing", False)
    config_params["gradient_accumulation_steps"] = hparams.get("accumulate_grad_batches", 1)
    config_params["mini_batch_size"] = hparams.get("per_device_train_mini_batch_size", 128)
    config_params["batch_size"] = hparams.get("per_device_train_batch_size", 128)

    config = PPOConfig(**config_params)

    ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer, dataset=train_dataset, data_collator=collator)

    # Move Reward Model to CUDA
    device = ppo_trainer.accelerator.device
    if ppo_trainer.accelerator.num_processes == 1:
        device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug
    reward_model.to(device)

    # output_length_sampler = LengthSampler(hparams.get("output_min_length", 16), hparams.get("output_max_length", 1024))

    generation_kwargs = {
        "num_beams": 1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": False,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "max_length": 8192,
        "max_new_tokens": 1024,
    }

    generation_kwargs = {
        "num_beams": 1,
        "do_sample": False,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 2048,
    }

    max_epochs = hparams.get("max_epochs", 999)
    max_steps = hparams.get("max_steps", -1)
    epoch_steps = len(ppo_trainer.dataloader)

    dataiter = iter(ppo_trainer.dataloader)

    for epoch in range(max_epochs):
        for step in enumerate(tqdm.tqdm(range(epoch_steps))):
            try:
                batch = next(dataiter)
            except StopIteration:
                dataiter = iter(ppo_trainer.dataloader)
                batch = next(dataiter)

            # dict_keys(['input_ids', 'attention_mask', 'labels', 'response'])
            query_tensor_input_ids = batch["input_ids"]
            query_tensors = [query_tensor for query_tensor in query_tensor_input_ids]

            response_tensors = ppo_trainer.generate(query_tensor=query_tensors, batch_size=2, return_prompt=True, **generation_kwargs)

            batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]
            for i in range(len(query_tensors)):
                print(tokenizer.decode(query_tensors[i].squeeze()))
                print("--------------")
                print(tokenizer.decode(response_tensors[i].squeeze()))
                print("===========")

            # Compute reward score
            encoded_texts = reward_tokenizer(batch["response"],
                padding="longest",
                truncation=True,
                return_tensors="pt",
                add_special_tokens=True,
            ).to(reward_model.device)

            rewards = reward_model.forward(
                input_ids=encoded_texts["input_ids"],
                attention_mask=encoded_texts["attention_mask"],
            )
            score_tensor = rewards.logits
            scores = [s.item() for s in score_tensor]
            # Run PPO step
            stats = ppo_trainer.step(query_tensors, response_tensors, scores)
            ppo_trainer.log_stats(stats, batch, scores)

            # Save Checkpoints
            # TODO: ....





