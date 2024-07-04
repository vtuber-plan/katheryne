# coding=utf-8
# Copyright 2024 XiaHan
# 
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import os
from typing import Any, Dict, Optional
import torch
from torch import nn, optim
from torch.nn import functional as F

from transformers import PreTrainedModel, get_scheduler

import lightning.pytorch as pl

from katheryne.utils.hparams import HParams
from katheryne.utils.model.model_utils import save_hf_format
from katheryne.utils.utils import get_optimizer_grouped_parameters, optimizer_to, save_zero_three_hf_model, save_zero_three_model

class RewardLanguageModel(pl.LightningModule):
    def __init__(self, model: PreTrainedModel, params: HParams, pad_token_id: Optional[int]=None) -> None:
        super().__init__()
        self.params = params

        if pad_token_id is not None:
            self.pad_token_id = pad_token_id
        else:
            self.pad_token_id = model.config.pad_token_id
        self.vocab_size = model.config.vocab_size

        self.model = model
        if model.config.pad_token_id is None:
            model.config.pad_token_id = self.pad_token_id

        self.deepspeed = self.params.get("strategy", None) == "deepspeed"
        self.strategy_params = self.params.get("strategy_params", dict())
        self.offload = self.strategy_params.get("offload_optimizer", False)

        self.num_padding_at_beginning = 0
        self.compute_fp32_loss = True

        self.save_hyperparameters(ignore=["model"])

        self.model_type = None
        if hasattr(model, "v_head"):
            self.model_type = "AutoModelForCausalLMWithValueHead"
        elif "SequenceClassification" in model.__class__.__name__:
            self.model_type = "AutoModelForSequenceClassification"
        elif "TokenClassification" in model.__class__.__name__:
            self.model_type = "AutoModelForTokenClassification"
        else:
            raise Exception("Unsupported Reward Model Type.")

    def _loss_forward_token_classification(self, tokens: Dict[str, torch.Tensor]):
        chosen_input_ids = tokens["chosen_input_ids"]
        chosen_attention_mask = tokens["chosen_attention_mask"]
        rejected_input_ids = tokens["rejected_input_ids"]
        rejected_attention_mask = tokens["rejected_attention_mask"]

        if chosen_input_ids is None or rejected_input_ids is None:
            raise Exception("The chosen_input_ids and rejected_input_ids shall not be None.")
        if chosen_input_ids is not None:
            chosen_batch_size = chosen_input_ids.shape[0]
        else:
            raise Exception("The chosen_input_ids shall not be None.")
        if rejected_input_ids is not None:
            rejected_batch_size = rejected_input_ids.shape[0]
        else:
            raise Exception("The rejected_input_ids shall not be None.")
        if chosen_batch_size != rejected_batch_size:
            raise Exception("The batch size of chosen sentences should equal to that of rejected sentences.")
        batch_size = chosen_batch_size
        if chosen_input_ids.shape[1] != rejected_input_ids.shape[1]:
            raise Exception("The sequence length of chosen_input_ids and rejected_input_ids shall be the same.")
        seq_len = chosen_input_ids.shape[1]

        # chosen_input_ids, rejected_input_ids: [batch, seq]
        input_ids = torch.cat([chosen_input_ids, rejected_input_ids], dim=0)
        attention_mask = torch.cat([chosen_attention_mask, rejected_attention_mask], dim=0)

        lm_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        # rewards: [bs, seq, 1]
        rewards = lm_outputs.logits
        # rewards: [bs, seq]
        rewards = rewards.squeeze(-1)

        chosen_ids = input_ids[:batch_size]  # bs x seq x 1
        rejected_ids = input_ids[batch_size:]
        chosen_rewards = rewards[:batch_size]
        rejected_rewards = rewards[batch_size:]

        # Compute pairwise loss. Only backprop on the different tokens before padding
        chosen_mean_scores = []
        rejected_mean_scores = []
        loss = 0.
        for i in range(batch_size):
            chosen_id = chosen_ids[i]
            rejected_id = rejected_ids[i]
            chosen_reward = chosen_rewards[i]
            rejected_reward = rejected_rewards[i]

            c_inds = (chosen_id == self.pad_token_id).nonzero()
            if len(c_inds) > self.num_padding_at_beginning:
                c_ind = c_inds[self.num_padding_at_beginning].item()
            else:
                c_ind = seq_len  # OPT model pads the first token, so we need to use the second padding token as the end of the sequence
            check_divergence = (chosen_id != rejected_id).nonzero()

            if len(check_divergence) == 0:
                end_ind = rejected_reward.size(-1)
                divergence_ind = end_ind - 1
                r_ind = c_ind
            else:
                # Check if there is any padding otherwise take length of sequence
                r_inds = (rejected_id == self.pad_token_id).nonzero()
                if len(r_inds) > self.num_padding_at_beginning:
                    r_ind = r_inds[self.num_padding_at_beginning].item()
                else:
                    r_ind = seq_len
                end_ind = max(c_ind, r_ind)
                divergence_ind = check_divergence[0]
            assert divergence_ind > 0
            c_truncated_reward = chosen_reward[divergence_ind:end_ind]
            r_truncated_reward = rejected_reward[divergence_ind:end_ind]
            chosen_mean_scores.append(chosen_reward[c_ind - 1])  #use the end score for reference
            rejected_mean_scores.append(rejected_reward[r_ind - 1])

            if self.compute_fp32_loss:
                c_truncated_reward = c_truncated_reward.float()
                r_truncated_reward = r_truncated_reward.float()
            loss += -torch.nn.functional.logsigmoid(c_truncated_reward - r_truncated_reward).mean()

        loss = loss / batch_size
        chosen_mean_scores = torch.stack(chosen_mean_scores)
        rejected_mean_scores = torch.stack(rejected_mean_scores)

        # , chosen_mean_scores, rejected_mean_scores
        return loss
    
    def _forward_token_classification(self,
                      input_ids=None,
                      attention_mask=None,
                      past_key_values=None,
                      head_mask=None,
                      inputs_embeds=None,
                      return_value_only=True,
                      prompt_length=0,
                      use_cache=False):
        if self.model.config.model_type == "llama":
            kwargs = dict()
        else:
            kwargs = dict(head_mask=head_mask)

        transformer_outputs = self.model(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs)
        values = transformer_outputs.logits.squeeze(-1)
        if return_value_only:
            return values
        else:
            # [0 0 0 0 prompt, answer, 0 0 0 0 ] for step 3, we have padding at the beginning
            # [prompt, answer, 0, 0, 0, 0] this is normal
            assert prompt_length > 1, "prompt_length must be greater than 1 to help select the end score"
            bs = values.size(0)
            seq_len = input_ids.shape[1]
            chosen_end_scores = [
            ]  # we use this name for consistency with the original forward function
            for i in range(bs):
                input_id = input_ids[i]
                value = values[i]

                c_inds = (input_id[prompt_length:] == self.pad_token_id).nonzero()
                # here we only use the answer part of the sequence so we do not need to care about the padding at the beginning
                c_ind = c_inds[0].item() + prompt_length if len(
                    c_inds) > 0 else seq_len
                chosen_end_scores.append(value[c_ind - 1])
            return {
                "values": values,
                "chosen_end_scores": torch.stack(chosen_end_scores),
            }

    def _loss_forward_sequence_classification(self, tokens: Dict[str, torch.Tensor]):
        chosen_input_ids = tokens["chosen_input_ids"]
        chosen_attention_mask = tokens["chosen_attention_mask"]
        rejected_input_ids = tokens["rejected_input_ids"]
        rejected_attention_mask = tokens["rejected_attention_mask"]

        if chosen_input_ids is None or rejected_input_ids is None:
            raise Exception("The chosen_input_ids and rejected_input_ids shall not be None.")
        if chosen_input_ids is not None:
            chosen_batch_size = chosen_input_ids.shape[0]
        else:
            raise Exception("The chosen_input_ids shall not be None.")
        if rejected_input_ids is not None:
            rejected_batch_size = rejected_input_ids.shape[0]
        else:
            raise Exception("The rejected_input_ids shall not be None.")
        if chosen_batch_size != rejected_batch_size:
            raise Exception("The batch size of chosen sentences should equal to that of rejected sentences.")
        batch_size = chosen_batch_size
        if chosen_input_ids.shape[1] != rejected_input_ids.shape[1]:
            raise Exception("The sequence length of chosen_input_ids and rejected_input_ids shall be the same.")
        seq_len = chosen_input_ids.shape[1]

        # chosen_input_ids, rejected_input_ids: [batch, seq]
        input_ids = torch.cat([chosen_input_ids, rejected_input_ids], dim=0)
        attention_mask = torch.cat([chosen_attention_mask, rejected_attention_mask], dim=0)

        lm_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        # rewards: [bs, 1]
        # (lm_logits, loss, value)
        rewards = lm_outputs.logits

        chosen_ids = input_ids[:batch_size]  # bs x seq x 1
        rejected_ids = input_ids[batch_size:]
        chosen_rewards = rewards[:batch_size]
        rejected_rewards = rewards[batch_size:]

        loss = -torch.nn.functional.logsigmoid(chosen_rewards - rejected_rewards).mean()

        # , chosen_rewards, rejected_rewards
        return loss

    def _forward_sequence_classification(self,
                input_ids=None,
                attention_mask=None,
                past_key_values=None
            ):
        lm_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        # rewards: [bs, 1]
        # (lm_logits, loss, value)
        lm_logits = lm_outputs.logits
        return lm_logits

    def _loss_forward_value_head_classification(self, tokens: Dict[str, torch.Tensor]):
        chosen_input_ids = tokens["chosen_input_ids"]
        chosen_attention_mask = tokens["chosen_attention_mask"]
        rejected_input_ids = tokens["rejected_input_ids"]
        rejected_attention_mask = tokens["rejected_attention_mask"]

        if chosen_input_ids is None or rejected_input_ids is None:
            raise Exception("The chosen_input_ids and rejected_input_ids shall not be None.")
        if chosen_input_ids is not None:
            chosen_batch_size = chosen_input_ids.shape[0]
        else:
            raise Exception("The chosen_input_ids shall not be None.")
        if rejected_input_ids is not None:
            rejected_batch_size = rejected_input_ids.shape[0]
        else:
            raise Exception("The rejected_input_ids shall not be None.")
        if chosen_batch_size != rejected_batch_size:
            raise Exception("The batch size of chosen sentences should equal to that of rejected sentences.")
        batch_size = chosen_batch_size
        if chosen_input_ids.shape[1] != rejected_input_ids.shape[1]:
            raise Exception("The sequence length of chosen_input_ids and rejected_input_ids shall be the same.")
        seq_len = chosen_input_ids.shape[1]

        # chosen_input_ids, rejected_input_ids: [batch, seq]
        input_ids = torch.cat([chosen_input_ids, rejected_input_ids], dim=0)
        attention_mask = torch.cat([chosen_attention_mask, rejected_attention_mask], dim=0)

        lm_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        # rewards: [bs, 1]
        # (lm_logits, loss, value)
        rewards = lm_outputs[0]

        chosen_ids = input_ids[:batch_size]  # bs x seq x 1
        rejected_ids = input_ids[batch_size:]
        chosen_rewards = rewards[:batch_size]
        rejected_rewards = rewards[batch_size:]

        loss = -torch.nn.functional.logsigmoid(chosen_rewards - rejected_rewards).mean()

        # , chosen_rewards, rejected_rewards
        return loss

    def _forward_value_head_classification(self,
                input_ids=None,
                attention_mask=None,
                past_key_values=None
            ):
        lm_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        # rewards: [bs, 1]
        # (lm_logits, loss, value)
        lm_logits = lm_outputs.logits
        return lm_logits

    def forward(self, input_ids=None,
                attention_mask=None,
                past_key_values=None,
                head_mask=None,
                inputs_embeds=None):
        if self.model_type == "AutoModelForCausalLMWithValueHead":
            return self._forward_value_head_classification(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
            )
        elif self.model_type == "AutoModelForSequenceClassification":
            return self._forward_sequence_classification(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
            )
        elif self.model_type == "AutoModelForTokenClassification":
            return self._forward_token_classification(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
            )
        else:
            raise Exception("Unsupported Reward Model Type.")

    def _loss_forward(self, tokens: Dict[str, torch.Tensor]):
        if self.model_type == "AutoModelForCausalLMWithValueHead":
            return self._loss_forward_value_head_classification(tokens)
        elif self.model_type == "AutoModelForSequenceClassification":
            return self._loss_forward_sequence_classification(tokens)
        elif self.model_type == "AutoModelForTokenClassification":
            return self._loss_forward_token_classification(tokens)
        else:
            raise Exception("Unsupported Reward Model Type.")

    def training_step(self, batch, batch_idx: int):
        chosen_input_ids = batch["chosen_input_ids"]
        chosen_attention_mask = batch["chosen_attention_mask"]
        rejected_input_ids = batch["rejected_input_ids"]
        rejected_attention_mask = batch["rejected_attention_mask"]

        batch_size = chosen_input_ids.shape[0]
        source_tokens = {
            'chosen_input_ids': chosen_input_ids,
            'chosen_attention_mask': chosen_attention_mask,
            'rejected_input_ids': rejected_input_ids,
            'rejected_attention_mask': rejected_attention_mask,
        }

        loss = self._loss_forward(tokens=source_tokens)

        # Logging to TensorBoard by default
        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True, rank_zero_only=True)
        return loss

    def validation_step(self, batch, batch_idx):
        chosen_input_ids = batch["chosen_input_ids"]
        chosen_attention_mask = batch["chosen_attention_mask"]
        rejected_input_ids = batch["rejected_input_ids"]
        rejected_attention_mask = batch["rejected_attention_mask"]

        batch_size = chosen_input_ids.shape[0]
        source_tokens = {
            'chosen_input_ids': chosen_input_ids,
            'chosen_attention_mask': chosen_attention_mask,
            'rejected_input_ids': rejected_input_ids,
            'rejected_attention_mask': rejected_attention_mask,
        }

        loss = self._loss_forward(tokens=source_tokens)

        self.log('val_loss', loss, on_step=True, on_epoch=True, sync_dist=True, rank_zero_only=True)

    def on_save_checkpoint(self, checkpoint):
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
        optimizer_grouped_parameters = get_optimizer_grouped_parameters(self.trainer.model, self.hparams.params.get("weight_decay", 0.0))
        if self.deepspeed:
            from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
            AdamOptimizer = DeepSpeedCPUAdam if self.offload else FusedAdam
            self.optim = AdamOptimizer(
                optimizer_grouped_parameters,
                lr=self.hparams.params.learning_rate,
                betas=self.hparams.params.betas,
                eps=self.hparams.params.eps
            )
        else:
            self.optim = torch.optim.Adam(
                optimizer_grouped_parameters, 
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
