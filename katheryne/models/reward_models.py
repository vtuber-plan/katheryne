# coding=utf-8
# Copyright 2024 XiaHan
# 
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from torch import nn, optim
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutputWithPast

class KatheryneForRewardModel(nn.Module):
    def __init__(self, base_model: PreTrainedModel, pad_token_id: int, num_padding_at_beginning=0, compute_fp32_loss=False):
        super().__init__()
        self.config = base_model.config
        self.base_model = base_model
        if hasattr(self.config, "word_embed_proj_dim"):
            # `OPT` models use word_embed_proj_dim as final output
            # https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_opt.py#L497
            self.v_head = nn.Linear(self.config.word_embed_proj_dim, 1, bias=False)
        else:
            # `gpt-neo(x)` models use `hidden_size` attribute names instead of `n_embd``
            self.config.n_embd = self.config.hidden_size if hasattr(self.config, "hidden_size") else self.config.n_embd
            self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)
        
        self.pad_token_id = pad_token_id
        print("====")
        print(self.pad_token_id)
        self.num_padding_at_beginning = num_padding_at_beginning
        self.compute_fp32_loss = compute_fp32_loss

    def get_input_embeddings(self):
        return self.base_model.embed_tokens

    def set_input_embeddings(self, value):
        self.base_model.embed_tokens = value

    def forward(
        self,
        chosen_input_ids: Optional[torch.LongTensor] = None,
        chosen_attention_mask: Optional[torch.Tensor] = None,
        rejected_input_ids: Optional[torch.LongTensor] = None,
        rejected_attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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

        lm_outputs = self.base_model(
            input_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = lm_outputs[0]
        # rewards: [bs, seq, 1]
        rewards = self.v_head(hidden_states)
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
            c_ind = c_inds[self.num_padding_at_beginning].item() if len(
                c_inds
            ) > self.num_padding_at_beginning else seq_len  # OPT model pads the first token, so we need to use the second padding token as the end of the sequence
            check_divergence = (chosen_id != rejected_id).nonzero()

            if len(check_divergence) == 0:
                end_ind = rejected_reward.size(-1)
                divergence_ind = end_ind - 1
                r_ind = c_ind
            else:
                # Check if there is any padding otherwise take length of sequence
                r_inds = (rejected_id == self.pad_token_id).nonzero()
                r_ind = r_inds[self.num_padding_at_beginning].item(
                ) if len(r_inds) > self.num_padding_at_beginning else seq_len
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

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
