# coding=utf-8
# Copyright 2024 XiaHan
# 
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""The base class of conversation-based datasets"""

from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import datasets
from torch.utils.data import Dataset

from transformers.trainer_pt_utils import LabelSmoother

from chatproto.conversation.history import ConversationHistory, ConversationSettings
from chatproto.registry import get_conv_settings

from katheryne.utils.model.tokenizer_utils import get_text_offset, is_merge_prefix_space, load_hf_tokenizer

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

class ConversationDataset(Dataset):
    def __init__(self, tokenizer_path: str, max_seq_len: int, pad_token_id: int,
                 conv_format: Union[str, ConversationSettings]="openbuddy",
                 end_of_conversation: Optional[Union[str, int]]=None) -> None:
        super().__init__()
        self.tokenizer_path = tokenizer_path
        self.tokenizer = None
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        if isinstance(conv_format, str):
            self.settings = get_conv_settings(conv_format)
        else:
            self.settings = conv_format
        
        self.end_of_conversation = end_of_conversation

        self.skip_space = False

    def tokenize(self, text: str, add_special_tokens: bool=True):
        encoded_text = self.tokenizer(text,
                        max_length=self.max_seq_len,
                        padding="longest",
                        truncation=True,
                        return_tensors="pt",
                        add_special_tokens=add_special_tokens,
                    )
        return encoded_text

    def add_end_of_conv(self, input_ids, attention_mask, end_of_conversation: Optional[Union[str, int]]):
        if end_of_conversation is None:
            return input_ids, attention_mask
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

        if self.end_of_conversation is not None:
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

    def get_prompt(self, messages, ignore_last:bool=False) -> Tuple[str, List[Tuple[int, int]]]:
        system = None
        for i, item in enumerate(messages):
            role, content = item["role"], item["content"]
            if role == "system":
                system = content
                break
        history = ConversationHistory(
            system=system,
            messages=[],
            offset=0,
            settings=self.settings,
        )

        for i, item in enumerate(messages):
            role, content = item["role"], item["content"]
            if role == "system":
                continue
            if ignore_last and i == len(messages) - 1:
                content = None
            if role.lower() == "user":
                real_role = self.settings.roles[0]
            elif role.lower() == "assistant":
                real_role = self.settings.roles[1]
            else:
                raise Exception(f"Unknown role {role}")
            history.messages.append((real_role, content))
        return history.get_prompt_and_indices()

    def _init_tokenizer(self):
        if self.tokenizer is None:
            self.tokenizer = load_hf_tokenizer(self.tokenizer_path, fast_tokenizer=True)
            self.skip_space = is_merge_prefix_space(self.tokenizer)

            if self.end_of_conversation is None:
                if self.tokenizer.eos_token_id is None:
                    self.end_of_conversation = self.tokenizer.pad_token_id
                else:
                    self.end_of_conversation = self.tokenizer.eos_token_id

