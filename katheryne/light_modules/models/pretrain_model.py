# coding=utf-8
# Copyright 2024 XiaHan
# 
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import os
from typing import Any, Dict
import torch
from torch import nn, optim
from torch.nn import functional as F

from transformers import PreTrainedModel, get_scheduler

import lightning.pytorch as pl

from katheryne.utils.hparams import HParams
from katheryne.utils.model.model_utils import save_hf_format
from katheryne.utils.utils import save_zero_three_hf_model, save_zero_three_model

class PretrainLanguageModel(pl.LightningModule):
    def __init__(self, model: PreTrainedModel, params: HParams) -> None:
        super().__init__()
        self.params = params

        self.model = model
        self.pad_token_id = self.model.config.pad_token_id
        self.vocab_size = self.model.config.vocab_size

        self.deepspeed = self.params.get("strategy", None) == "deepspeed"
        self.strategy_params = self.params.get("strategy_params", dict())
        self.offload = self.strategy_params.get("offload_optimizer", False)

        self.save_hyperparameters(ignore=["model"])

    def forward(self, tokens: Dict[str, torch.Tensor]):
        input_ids, input_mask = tokens["input_ids"], tokens["attention_mask"]
        batch_size = input_ids.shape[0]

        lm_output = self.model(
            input_ids=input_ids,
            attention_mask=input_mask,
            labels=input_ids,
            use_cache=False,
            return_dict=True,
        )
        return lm_output

    def training_step(self, batch, batch_idx: int):
        input_ids, input_mask = batch["input_ids"], batch["attention_mask"]

        batch_size = input_ids.shape[0]
        source_tokens = {
            'input_ids': input_ids,
            'attention_mask': input_mask
        }

        lm_output = self.forward(tokens=source_tokens)
        loss = lm_output[0]

        # Logging to TensorBoard by default
        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, input_mask = batch["input_ids"], batch["attention_mask"]

        batch_size = input_ids.shape[0]
        source_tokens = {
            'input_ids': input_ids,
            'attention_mask': input_mask
        }

        lm_output = self.forward(tokens=source_tokens)
        loss = lm_output.loss

        self.log('val_loss', loss, on_step=True, on_epoch=True, sync_dist=True)

    def on_save_checkpoint(self, checkpoint):
        if self.trainer.logger is None:
            return
        save_path = f"{self.trainer.logger.log_dir}/huggingface_format"
        if self.deepspeed and self.strategy_params.get("zero_stage", 0) == 3:
            # For zero stage 3, each gpu only has a part of the model, so we need a special save function
            save_zero_three_hf_model(self.model, self.global_rank, 
                                  os.path.join(save_path, f"checkpoint-step-{self.global_step}"),
                                  zero_stage=3
                                )
        else:
            if self.global_rank == 0:
                save_hf_format(
                    self.model,
                    tokenizer=None,
                    output_dir=save_path,
                    sub_folder=f"checkpoint-step-{self.global_step}",
                    peft_merge=self.hparams.get("peft_merge", False),
                )
    
    def configure_optimizers(self):
        if self.deepspeed:
            from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
            AdamOptimizer = DeepSpeedCPUAdam if self.offload else FusedAdam
            self.optim = AdamOptimizer(
                self.trainer.model.parameters(),
                lr=self.hparams.params.learning_rate,
                betas=self.hparams.params.betas,
                eps=self.hparams.params.eps
            )
        else:
            self.optim = torch.optim.Adam(
                self.trainer.model.parameters(), 
                self.hparams.params.learning_rate, 
                betas=self.hparams.params.betas, 
                eps=self.hparams.params.eps
            )
        
        stepping_batches = self.trainer.estimated_stepping_batches
        self.scheduler = get_scheduler(
            name=self.hparams.params.lr_scheduler_type,
            optimizer=self.optim,
            num_warmup_steps=self.hparams.params.get("num_warmup_steps", 0),
            num_training_steps=stepping_batches,
        )

        return {
            "optimizer": self.optim,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
