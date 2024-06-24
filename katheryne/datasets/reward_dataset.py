# coding=utf-8
# Copyright 2024 XiaHan
# 
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""RLHF Reward Model datasets"""
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import datasets
from torch.utils.data import Dataset

from transformers.trainer_pt_utils import LabelSmoother

from chatproto.conversation.history import ConversationHistory, ConversationSettings
from chatproto.registry import get_conv_settings

from katheryne.utils.model.tokenizer_utils import get_text_offset, is_merge_prefix_space, load_hf_tokenizer

from .conversation_dataset import ConversationDataset

class RewardDataset(ConversationDataset):
    def __init__(self, dataset: datasets.Dataset, tokenizer_path: str, max_seq_len: int,
                 pad_token_id: int,
                 conv_format: Union[str, ConversationSettings]="openbuddy",
                 end_of_conversation: Optional[Union[str, int]]=None
                ) -> None:
        super().__init__(tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
            pad_token_id=pad_token_id,
            conv_format=conv_format,
            end_of_conversation=end_of_conversation
        )
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        self._init_tokenizer()
        sample = self.dataset[idx]

        messages = sample["messages"]
        chosen = sample["chosen"]
        rejected = sample["rejected"]

        chosen_messages = messages + {"role": "assistant", "content": chosen}
        rejected_messages = messages + {"role": "assistant", "content": rejected}

        """Chosen messages"""
        chosen_prompt, chosen_indices = self.get_prompt(chosen_messages)
        if isinstance(self.end_of_conversation, str):
            chosen_prompt += self.end_of_conversation
        
        chosen_encoded_text = self.tokenize(chosen_prompt, add_special_tokens=True)
        chosen_input_ids = chosen_encoded_text["input_ids"].squeeze(0)
        chosen_attention_mask = chosen_encoded_text["attention_mask"].squeeze(0)

        # if truncation not work
        if len(chosen_input_ids) > self.max_seq_len:
            chosen_input_ids = chosen_input_ids[:self.max_seq_len]
            chosen_input_ids[-1] = self.tokenizer.eos_token_id
            chosen_attention_mask = chosen_attention_mask[:self.max_seq_len]
        
        chosen_input_ids, chosen_attention_mask = self.add_end_of_conv(chosen_input_ids, chosen_attention_mask, self.end_of_conversation)

        """Rejected messages"""
        rejected_prompt, rejected_indices = self.get_prompt(rejected_messages)
        if isinstance(self.end_of_conversation, str):
            rejected_prompt += self.end_of_conversation

        rejected_encoded_text = self.tokenize(rejected_prompt, add_special_tokens=True)
        rejected_input_ids = rejected_encoded_text["input_ids"].squeeze(0)
        rejected_attention_mask = rejected_encoded_text["attention_mask"].squeeze(0)

        # if truncation not work
        if len(rejected_input_ids) > self.max_seq_len:
            rejected_input_ids = rejected_input_ids[:self.max_seq_len]
            rejected_input_ids[-1] = self.tokenizer.eos_token_id
            rejected_attention_mask = rejected_attention_mask[:self.max_seq_len]
        
        rejected_input_ids, rejected_attention_mask = self.add_end_of_conv(rejected_input_ids, rejected_attention_mask, self.end_of_conversation)


        return {
            "chosen_input_ids": chosen_input_ids,
            "chosen_attention_mask": chosen_attention_mask,
            "rejected_input_ids": rejected_input_ids,
            "rejected_attention_mask": rejected_attention_mask,
        }
