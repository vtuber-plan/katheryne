# coding=utf-8
# Copyright 2024 XiaHan
# 
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import torch
from torch.utils.data import Dataset, Subset, ConcatDataset
import datasets
from katheryne.utils.model.tokenizer_utils import load_hf_tokenizer


class PretrainDataset(Dataset):
    def __init__(self, pretrain_dataset: datasets.Dataset, tokenizer_path: str, max_seq_len: int, pad_token_id: int) -> None:
        super().__init__()
        self.pretrain_dataset = pretrain_dataset
        self.tokenizer_path = tokenizer_path
        self.tokenizer = None
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id

    def __len__(self):
        length = len(self.pretrain_dataset)
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

    def __getitem__(self, idx):
        if self.tokenizer is None:
            self.tokenizer = load_hf_tokenizer(self.tokenizer_path, fast_tokenizer=True)
        encoded_text = self.tokenize(self.pretrain_dataset[idx]["text"])
        return {
            "input_ids": encoded_text["input_ids"].squeeze(0),
            "attention_mask": encoded_text["attention_mask"].squeeze(0),
            "labels": encoded_text["input_ids"].squeeze(0),
        }
