import sys
import os
import math
from typing import Tuple
import json
import glob
import argparse
import platform

from katheryne.light_modules.utils.checkpoints import get_lastest_checkpoint
from katheryne.light_modules.models.pretrain_model import PretrainLanguageModel
from katheryne.utils.hparams import HParams
from katheryne.data.instruction import create_instruction_dataset
from katheryne.data.collators import DataCollatorWithPadding

from katheryne.utils.ds_utils import get_train_ds_config
from katheryne.utils.model.model_utils import create_hf_model, save_hf_format
from katheryne.utils.model.tokenizer_utils import load_hf_tokenizer
from katheryne.utils.utils import get_optimizer_grouped_parameters
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

import tqdm

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
torch.set_float32_matmul_precision('medium')

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, GradientAccumulationScheduler
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.profilers import SimpleProfiler, AdvancedProfiler

import lightning_fabric

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizerBase,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)

def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain a transformers model on a causal language modeling task")
    parser.add_argument('--hparams', type=str, default="hparams/hparams_llama2_7b_lora.json", help='The hparam file of training')
    parser.add_argument('--accelerator', type=str, default="gpu", help='training device')
    parser.add_argument('--device', type=str, default="0,1", help='training device ids')
    parser.add_argument('--checkpoint', type=str, default="checkpoints/", help='checkpoint path')
    parser.add_argument('--seed', type=int, default=43, help='model seed')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    hparams = HParams.from_json_file(args.hparams)
    master_port = os.environ.get("MASTER_PORT", None)
    master_addr = os.environ.get("MASTER_ADDR", None)
    world_size = os.environ.get("WORLD_SIZE", None)
    rank = os.environ.get("RANK", None)

    # Validate settings
    if hparams.get("gradient_checkpointing", False) and hparams.get("lora_dim", 0) > 0:
        assert (
            not hparams.get("only_optimize_lora", True)
        ), "--gradient_checkpointing and --only_optimize_lora cannot be enabled at the same time."

    # If passed along, set the training seed now.
    lightning_fabric.seed_everything(args.seed)

    # Load tokenizer and model
    tokenizer = load_hf_tokenizer(hparams.model_name_or_path, fast_tokenizer=True)
    if tokenizer._pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype_str = hparams.get("model_torch_dtype", "auto")
    if torch_dtype_str in ["bfloat16", "bf16"]:
        torch_dtype = torch.bfloat16
    elif torch_dtype_str in ["half", "float16", "fp16"]:
        torch_dtype = torch.half
    else:
        torch_dtype = "auto"
    model = create_hf_model(
        model_class=AutoModelForCausalLM,
        model_name_or_path=hparams.model_name_or_path,
        tokenizer=tokenizer,
        dtype=torch_dtype,
        disable_dropout=hparams.disable_dropout
    )

    # Setup LORA
    if hparams.get("lora_dim", 0) > 0:
        from peft import LoftQConfig, LoraConfig, get_peft_model
        base_model = model
        # loftq_config = LoftQConfig(loftq_bits=4, ...)           # set 4bit quantization
        lora_config = LoraConfig(
            r=hparams.get("r", 128),
            target_modules=hparams.get("target_modules", 8),
            lora_alpha=hparams.get("lora_alpha", 8),
            lora_dropout=hparams.get("lora_dropout", 0.0),
            fan_in_fan_out=hparams.get("fan_in_fan_out", False),
            bias=hparams.get("bias", 'None'),
        )
        model = get_peft_model(base_model, lora_config)
    
    if hparams.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
    
    # Save Model
    save_hf_format(model, tokenizer, "./lightning_logs", "huggingface_format")
    
    # Prepare the data
    print("***** Prepare Dataset *****")
    train_dataset, valid_dataset = create_pretrain_dataset(
        hparams.data_path,
        hparams.data_output_path,
        args.seed,
        tokenizer,
        hparams.max_seq_len
    )

    # DataLoaders creation:
    print("***** DataLoaders creation *****")
    train_sampler = RandomSampler(train_dataset)
    valid_sampler = SequentialSampler(valid_dataset)
    
    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest", max_length=hparams.max_seq_len)
    train_dataloader = DataLoader(train_dataset, collate_fn=collator, sampler=train_sampler, num_workers=4, batch_size=hparams.per_device_train_batch_size)
    valid_dataloader = DataLoader(valid_dataset, collate_fn=collator, sampler=valid_sampler, num_workers=4, batch_size=hparams.per_device_eval_batch_size)

    model = PretrainLanguageModel(
        model,
        hparams,
    )

    # Checkpoint Settings
    checkpoint_every_n_train_steps = 100
    if "checkpoint_every_n_train_steps" in hparams:
        checkpoint_every_n_train_steps = hparams.checkpoint_every_n_train_steps
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=None, save_last=True, every_n_train_steps=checkpoint_every_n_train_steps, save_weights_only=False, save_on_train_epoch_end=True
    )

    # Earlystop Settings
    # monitor="val_loss", mode="min", save_top_k=5
    # earlystop_callback = EarlyStopping(monitor="valid/loss_mel_epoch", mode="min", patience=13)

    # Leaning rate monitor
    learning_rate_callback = LearningRateMonitor(logging_interval="step")

    # GradientAccumulationScheduler
    # accumulator_callback = GradientAccumulationScheduler(scheduling={4: 2})

    devices = [int(n.strip()) for n in args.device.split(",")]
    trainer_params = {
        "accelerator": args.accelerator,
        "callbacks": [checkpoint_callback, learning_rate_callback],
    }

    # Logger Settings
    trainer_params["log_every_n_steps"] = hparams.get("log_every_n_steps", 50)
    trainer_params["val_check_interval"] = hparams.get("val_check_interval", 1.0)

    # Devices
    if args.accelerator != "cpu":
        trainer_params["devices"] = devices

    if platform.system().lower() == 'windows':
        backend = "gloo"
    else:
        backend = "nccl"

    if "fp16" in hparams and hparams.fp16:
        print("using fp16")
        trainer_params["precision"] = "16-mixed"
        ds_precision = "fp16"
    elif "bf16" in hparams and hparams.bf16:
        print("using bf16")
        trainer_params["precision"] = "bf16-mixed"
        ds_precision = "bf16"
    else:
        ds_precision = "fp32"

    if "strategy" in hparams:
        if hparams.strategy == "fsdp":
            from lightning.pytorch.strategies import FSDPStrategy
            fsdp = FSDPStrategy(
                cpu_offload=False,
                process_group_backend=backend
            )
            trainer_params["strategy"] = fsdp
        elif hparams.strategy == "deepspeed":
            if world_size is None:
                ds_world_size = len(devices)
            else:
                ds_world_size = int(world_size)

            from lightning.pytorch.strategies import DeepSpeedStrategy
            ds_config = get_train_ds_config(
                offload=hparams.offload,
                stage=hparams.zero_stage,
                precision=ds_precision
            )
            ds_config['train_micro_batch_size_per_gpu'] = hparams.per_device_train_batch_size
            ds_config['train_batch_size'] = hparams.per_device_train_batch_size * ds_world_size * hparams.accumulate_grad_batches
            ds = DeepSpeedStrategy(
                zero_optimization=True,
                stage=hparams.zero_stage,
                remote_device = hparams.get("remote_device", "cpu"),
                offload_optimizer = hparams.offload,
                offload_optimizer_device = 'cpu',
                offload_parameters = hparams.offload,
                cpu_checkpointing = hparams.offload,
                offload_params_device = "cpu",
                nvme_path=hparams.get("nvme_path", "./nvme_offload"),
                contiguous_memory_optimization=True,
                config=ds_config,
            )
            trainer_params["strategy"] = ds
        elif hparams.strategy == "ddp":
            ddp = DDPStrategy(process_group_backend=backend, find_unused_parameters=True)
            trainer_params["strategy"] = ddp
    elif len(devices) > 1:
        ddp = DDPStrategy(process_group_backend=backend, find_unused_parameters=True)
        trainer_params["strategy"] = ddp

    trainer_params["max_epochs"] = hparams.get("max_epochs", 1000)
    trainer_params["accumulate_grad_batches"] = hparams.get("accumulate_grad_batches", 1)
    # profiler = AdvancedProfiler(filename="profile.txt")

    # Other params
    if "trainer" in hparams and isinstance(hparams.trainer, dict):
        for key, value in hparams.trainer:
            trainer_params[key] = value
    
    trainer = pl.Trainer(**trainer_params) # , profiler=profiler, max_steps=200
    # Resume training
    ckpt_path = get_lastest_checkpoint("./lightning_logs", "checkpoints")

    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader, ckpt_path=ckpt_path)

if __name__ == "__main__":
    main()
