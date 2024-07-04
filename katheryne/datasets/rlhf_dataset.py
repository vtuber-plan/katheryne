# coding=utf-8
# Copyright 2024 XiaHan
# 
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Chat-based RLHF datasets"""
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

class RLHFDataset(ConversationDataset):
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
        if len(messages) > 0:
            if messages[-1]["role"].lower() == "assistant":
                messages[-1]["content"] = None
            elif messages[-1]["role"].lower() == "user":
                messages.append({
                    "role": "assistant",
                    "content": None,
                })
        prompt, indices = self.get_prompt(messages)
        # if isinstance(self.end_of_conversation, str):
        #     prompt += self.end_of_conversation

        encoded_text = self.tokenize(prompt, add_special_tokens=True)

        input_ids = encoded_text["input_ids"].squeeze(0)
        attention_mask = encoded_text["attention_mask"].squeeze(0)

        # if truncation not work
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
            input_ids[-1] = self.tokenizer.eos_token_id
            attention_mask = attention_mask[:self.max_seq_len]

        # input_ids, attention_mask = self.add_end_of_conv(input_ids, attention_mask, self.end_of_conversation)

        labels = input_ids.clone()
        labels = self.mask_label(prompt, labels, indices)
        # print(len(input_ids), len(labels))
        # TODO: labels pad上IGNORE_TOKEN_ID
        # labels[:len(encoded_prompt) + 1] = IGNORE_TOKEN_ID # 这里不 + 1抵消bos，是因为可能最后一个token是空格，和回答的第一个token合在一起
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
