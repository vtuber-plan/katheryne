# coding=utf-8
# Copyright 2024 XiaHan
# 
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, Subset, ConcatDataset
import datasets

from transformers.trainer_pt_utils import LabelSmoother

from katheryne.utils.model.tokenizer_utils import get_text_offset, is_merge_prefix_space, load_hf_tokenizer
from .conversation_dataset import ConversationDataset

from chatproto.conversation.history import ConversationHistory, ConversationSettings
from chatproto.registry import get_conv_settings

class InstructionDataset(ConversationDataset):
    def __init__(self, dataset: datasets.Dataset,
                 tokenizer_path: str, max_seq_len: int,
                 pad_token_id: int,
                 conv_format: str="openbuddy",
                 end_of_conversation: Optional[Union[str, int]]=None,
                 system_prompt: Optional[str]=None,
                 instruction_prompt: Optional[str]=None) -> None:
        super().__init__(tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
            pad_token_id=pad_token_id,
            conv_format=conv_format,
            end_of_conversation=end_of_conversation
        )
        self.dataset = dataset

        if isinstance(conv_format, str):
            self.settings = get_conv_settings(conv_format)
        else:
            self.settings = conv_format
        
        self.end_of_conversation = end_of_conversation
        
        if system_prompt is None:
            self.system_prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."
        else:
            self.system_prompt = system_prompt
        
        if instruction_prompt is None:
            self.instruction_prompt = "Instruction:\n{Instruction}\nInput:\n{Input}\n"
        else:
            self.instruction_prompt = instruction_prompt

    def __getitem__(self, idx):
        self._init_tokenizer()
        sample = self.dataset[idx]

        instruction = self.dataset[idx]["instruction"]
        input = self.dataset[idx]["input"]
        output = self.dataset[idx]["output"]

        prompt, indices = self.get_prompt(instruction, input, output)
        if isinstance(self.end_of_conversation, str):
            prompt += self.end_of_conversation

        encoded_text = self.tokenize(prompt, add_special_tokens=True)

        input_ids = encoded_text["input_ids"].squeeze(0)
        attention_mask = encoded_text["attention_mask"].squeeze(0)

        # if truncation not work
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
            input_ids[-1] = self.tokenizer.eos_token_id
            attention_mask = attention_mask[:self.max_seq_len]

        input_ids, attention_mask = self.add_end_of_conv(input_ids, attention_mask, self.end_of_conversation)

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
