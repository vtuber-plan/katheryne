

import torch
from torch.utils.data import Dataset, Subset, ConcatDataset

from transformers import PreTrainedTokenizerBase

from transformers.trainer_pt_utils import LabelSmoother
IGNORE_TOKEN_ID = LabelSmoother.ignore_index

class InstructionDataset(Dataset):

    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_seq_len: int, dataset: PromptUniformDataset, pad_token_id: int) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.dataset = dataset
        self.pad_token_id = pad_token_id

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
        encoded_prompt = self.tokenize(sample["prompt_dataset"], add_special_tokens=False)
        encoded_text = self.tokenize(sample["chosen_dataset"])
        labels = encoded_text["input_ids"].squeeze(0).clone()
        labels[:len(encoded_prompt) + 1] = IGNORE_TOKEN_ID # 这里不 + 1抵消bos，是因为可能最后一个token是空格，和回答的第一个token合在一起
        return {
            "input_ids": encoded_text["input_ids"].squeeze(0),
            "attention_mask": encoded_text["attention_mask"].squeeze(0),
            "labels": labels
        }
