

import torch
from torch.utils.data import Dataset, Subset, ConcatDataset

from transformers import PreTrainedTokenizerBase

from transformers.trainer_pt_utils import LabelSmoother
IGNORE_TOKEN_ID = LabelSmoother.ignore_index

class PretrainUniformDataset(Dataset):

    def __init__(self, pretrain_dataset) -> None:
        super().__init__()
        self.pretrain_dataset = pretrain_dataset

    def __len__(self):
        length = len(self.pretrain_dataset)
        return length

    def __getitem__(self, idx):
        return self.pretrain_dataset[idx]

class PretrainDataset(Dataset):

    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_seq_len: int, pretrain_dataset: PretrainUniformDataset, pad_token_id: int) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pretrain_dataset = pretrain_dataset
        self.pad_token_id = pad_token_id

    def __len__(self):
        length = len(self.pretrain_dataset)
        return length
    
    def tokenize(self, text):
        encoded_text = self.tokenizer(text,
                        max_length=self.max_seq_len,
                        truncation=True,
                        return_tensors="pt"
                    )
        return encoded_text

    def __getitem__(self, idx):
        encoded_text = self.tokenize(self.pretrain_dataset[idx]["text"])
        return {
            "input_ids": encoded_text["input_ids"].squeeze(0),
            "attention_mask": encoded_text["attention_mask"].squeeze(0),
            "labels": encoded_text["input_ids"].squeeze(0),
        }


class PromptUniformDataset(Dataset):

    def __init__(self, prompt_dataset, chosen_dataset, reject_dataset, train_phase) -> None:
        super().__init__()
        self.prompt_dataset = prompt_dataset
        self.chosen_dataset = chosen_dataset
        self.reject_dataset = reject_dataset
        self.train_phase = train_phase

    def __len__(self):
        length = len(self.chosen_dataset)
        if self.train_phase == 3:
            length = len(self.prompt_dataset)
        return length

    def __getitem__(self, idx):
        return {
            "prompt_dataset": self.prompt_dataset[idx],
            "chosen_dataset": self.chosen_dataset[idx],
            "reject_dataset": self.reject_dataset[idx],
        }


class PromptDataset(Dataset):

    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_seq_len: int, dataset: PromptUniformDataset, pad_token_id: int, train_phase: int) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.dataset = dataset
        self.pad_token_id = pad_token_id
        self.train_phase = train_phase

    def __len__(self):
        return len(self.dataset)

    def tokenize(self, text: str, add_special_tokens: bool=True):
        encoded_text = self.tokenizer(text,
                        max_length=self.max_seq_len,
                        truncation=True,
                        return_tensors="pt",
                        add_special_tokens=add_special_tokens,
                    )
        return encoded_text

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        if self.train_phase == 1:
            encoded_prompt = self.tokenize(sample["prompt_dataset"], add_special_tokens=False)
            encoded_text = self.tokenize(sample["chosen_dataset"])
            labels = encoded_text["input_ids"].squeeze(0).clone()
            labels[:len(encoded_prompt) + 1] = IGNORE_TOKEN_ID # 这里不 + 1抵消bos，是因为可能最后一个token是空格，和回答的第一个token合在一起
            return {
                "input_ids": encoded_text["input_ids"].squeeze(0),
                "attention_mask": encoded_text["attention_mask"].squeeze(0),
                "labels": labels
            }
        elif self.train_phase == 2:
            encoded_chosen = self.tokenize(sample["chosen_dataset"])
            encoded_reject = self.tokenize(sample["reject_dataset"])
            return {
                "chosen_input_ids": encoded_chosen["input_ids"].squeeze(0),
                "chosen_attention_mask": encoded_chosen["attention_mask"].squeeze(0),
                "reject_input_ids": encoded_reject["input_ids"].squeeze(0),
                "reject_attention_mask": encoded_reject["attention_mask"].squeeze(0),
            }
        elif self.train_phase == 3:
            encoded_prompt = self.tokenize(sample["prompt_dataset"])
            return {
                "input_ids": encoded_prompt["input_ids"].squeeze(0),
                "attention_mask": encoded_prompt["attention_mask"].squeeze(0),
                "pad_token_id": self.pad_token_id,
            }
