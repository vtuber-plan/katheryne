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

from chatproto.conversation.history import ConversationHistory, ConversationSettings
from chatproto.registry import get_conv_settings

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

class InstructionDataset(Dataset):
    def __init__(self, tokenizer_path: str, max_seq_len: int,
                 dataset: datasets.Dataset, pad_token_id: int,
                 conv_format: str="openbuddy",
                 end_of_conversation: Optional[Union[str, int]]=None,
                 system_prompt: Optional[str]=None,
                 instruction_prompt: Optional[str]=None) -> None:
        super().__init__()
        self.tokenizer_path = tokenizer_path
        self.tokenizer = None
        self.max_seq_len = max_seq_len
        self.dataset = dataset
        self.pad_token_id = pad_token_id

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

    def __len__(self):
        length = len(self.dataset)
        return length

    def tokenize(self, text: str, add_special_tokens: bool=True):
        encoded_text = self.tokenizer(text,
                        max_length=self.max_seq_len,
                        padding="longest",
                        truncation=True,
                        return_tensors="pt",
                        add_special_tokens=add_special_tokens,
                    )
        return encoded_text

    def add_end_of_conv(self, input_ids, attention_mask, end_of_conversation: Union[str, int]):
        if isinstance(end_of_conversation, int):
            last_token_id = input_ids[-1]
            if last_token_id == self.tokenizer.eos_token_id:
                if end_of_conversation != self.tokenizer.eos_token_id:
                    input_ids[-1] = end_of_conversation
                    input_ids = torch.cat((
                            input_ids,
                            torch.tensor([self.tokenizer.eos_token_id], dtype=torch.long, device=input_ids.device)
                        ), dim=0)
                    attention_mask = torch.cat((
                        attention_mask,
                        torch.tensor([1], dtype=torch.long, device=attention_mask.device)
                        ), dim=0)
            else:
                input_ids = torch.cat((
                        input_ids,
                        torch.tensor([end_of_conversation], dtype=torch.long, device=input_ids.device)
                    ), dim=0)
                attention_mask = torch.cat((
                    attention_mask,
                    torch.tensor([1], dtype=torch.long, device=attention_mask.device)
                    ), dim=0)
        return input_ids, attention_mask

    
    def mask_label(self, prompt: str, target: torch.Tensor, indices: List[Tuple[int, int]]):
        tokens = self.tokenizer.convert_ids_to_tokens(target, skip_special_tokens=False)
        text_offset = get_text_offset(self.tokenizer, prompt, tokens, has_special_tokens=True)

        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID
        start = cur_len
        end = cur_len
        for i, turn in enumerate(indices[1:]):
            if i % 2 == 0:
                continue
            text_start, text_end = turn
            if prompt[text_start - 1] == " " and self.skip_space:
                text_start -= 1
            start = np.searchsorted(text_offset, text_start, side="left")
            end = np.searchsorted(text_offset, text_end, side="left")
            target[cur_len:start] = IGNORE_TOKEN_ID
            cur_len = end

        if isinstance(self.end_of_conversation, str):
            end_conv = np.searchsorted(text_offset, len(prompt)-len(self.end_of_conversation))
        elif isinstance(self.end_of_conversation, int):
            end_conv = np.searchsorted(text_offset, len(prompt)-1)
        else:
             raise Exception(f"Type of end_of_conversation is {type(self.end_of_conversation)}, which is not supported.")
        target[end_conv:end] = IGNORE_TOKEN_ID
        if False:  # Inspect and check the correctness of masking
            z = target.clone()
            if self.tokenizer.unk_token_id is not None:
                z = torch.where(z == IGNORE_TOKEN_ID, self.tokenizer.unk_token_id, z)
            else:
                z = torch.where(z == IGNORE_TOKEN_ID, 0, z)
            print(prompt)
            print("================")
            print(self.tokenizer.decode(z))
            exit()
        return target

    def get_prompt(self, instruction, input, output, ignore_last:bool=False) -> Tuple[str, List[Tuple[int, int]]]:
        history = ConversationHistory(
            system=self.system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": self.instruction_prompt.replace("{Instruction}", instruction).replace("{Input}", input)
                },
                {
                    "role": "assistant",
                    "content": output
                },
            ],
            offset=0,
            settings=self.settings,
        )
        prompt, indices = history.get_prompt_and_indices()
        return prompt, indices

    def __getitem__(self, idx):
        if self.tokenizer is None:
            self.tokenizer = load_hf_tokenizer(self.tokenizer_path, fast_tokenizer=True)
            self.skip_space = is_merge_prefix_space(self.tokenizer)
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
