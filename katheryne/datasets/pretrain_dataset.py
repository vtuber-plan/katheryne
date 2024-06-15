
import torch
from torch.utils.data import Dataset, Subset, ConcatDataset
import datasets
from transformers import PreTrainedTokenizerBase


class PretrainDataset(Dataset):

    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_seq_len: int, pretrain_dataset: datasets.Dataset, pad_token_id: int) -> None:
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
                        padding="longest",
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
