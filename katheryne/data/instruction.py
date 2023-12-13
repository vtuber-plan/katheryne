

import hashlib
import os
import numpy as np
import pyarrow as pa

import torch
from torch.utils.data import Dataset, Subset, ConcatDataset
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase
from katheryne.utils.utils import chunked

from katheryne.utils.data.data_utils import get_shuffle_idx
from katheryne.utils.diskist import write_diskist, Diskist

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


class SharegptCleanedDataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "Vtuber-plan/sharegpt-cleaned"
        self.dataset_name_clean = "Vtuber_plan_sharegpt_cleaned"
        self.sep = "\n### "

    def get_train_data(self):
        if "val" in self.raw_datasets:
            dataset = self.raw_datasets["train"]
        else:
            dataset = self.raw_datasets["train"]
            index_list = get_subset(self.seed, dataset, [0.9, 0.1])
            index = index_list[0]
            dataset = Subset(dataset, index)
            return dataset

    def get_eval_data(self):
        if "val" in self.raw_datasets:
            dataset = self.raw_datasets["val"]
        else:
            dataset = self.raw_datasets["train"]
            index_list = get_subset(self.seed, dataset, [0.9, 0.1])
            index = index_list[1]
            dataset = Subset(dataset, index)
        return dataset

    def get_prompt(self, sample):
        messages = sample["messages"]

        settings = get_conv_settings("ningyu")
        system = "A chat between a curious human and an artificial intelligence assistant. "
        for i, item in enumerate(messages):
            role, content = item["role"], item["content"]
            if role == "system":
                system = content
                break
        history = ConversationHistory(
            system=system,
            messages=[],
            offset=0,
            settings=settings,
        )

        for i, item in enumerate(messages):
            role, content = item["role"], item["content"]
            if role == "system":
                continue
            if i == len(messages) - 1:
                content = None
            history.messages.append((role, content))
        return history.get_prompt()

    def get_chosen(self, sample):
        messages = sample["messages"]
        return messages[-1]["content"]

    def get_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None

    def get_prompt_and_chosen(self, sample):
        messages = sample["messages"]

        settings = get_conv_settings("ningyu")
        system = "A chat between a curious human and an artificial intelligence assistant. "
        for i, item in enumerate(messages):
            role, content = item["role"], item["content"]
            if role == "system":
                system = content
                break
        history = ConversationHistory(
            system=system,
            messages=[],
            offset=0,
            settings=settings,
        )

        for i, item in enumerate(messages):
            role, content = item["role"], item["content"]
            if role == "system":
                continue
            history.messages.append((role, content))
        return history.get_prompt()

    def get_prompt_and_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None


def create_dataset(dataset_name, output_path,
                   seed, tokenizer, end_of_conversation_token,
                   max_seq_len):
    raw_dataset = get_raw_dataset(dataset_name, output_path, seed)
    train_dataset = raw_dataset.get_train_data()
    train_dataset = create_uniform_dataset(train_dataset, raw_dataset, tokenizer, end_of_conversation_token, max_seq_len)
    eval_dataset = raw_dataset.get_eval_data()
    eval_dataset = create_uniform_dataset(eval_dataset, raw_dataset, tokenizer, end_of_conversation_token, max_seq_len)
    return train_dataset, eval_dataset

def create_supervised_finetuning_dataset(data_path, output_path, seed,
                            tokenizer, max_seq_len, end_of_conversation_token="<|endoftext|>"):
    """
    Creates the sft dataset
    """
    os.makedirs(output_path, exist_ok=True)
    fname = "_".join(data_path)
    tokenizer_name = tokenizer.init_kwargs["name_or_path"].replace("/", "_")
    fname = f"{fname}_seed{seed}_tokenizer{tokenizer_name}_seqlen{max_seq_len}"
    fname = "_".join(fname.split("/"))
    fname = hashlib.sha256(fname.encode()).hexdigest()  # hash the file name to avoid too long file name
    train_fname = f"{output_path}/traindata_{fname}.diskist"
    eval_fname = f"{output_path}/evaldata_{fname}.diskist"

    # cache_found = os.path.isfile(train_fname) and os.path.isfile(eval_fname)
    cache_found = os.path.isdir(train_fname) and os.path.isdir(eval_fname)
    buf_create_cache = torch.ByteTensor([not cache_found]).cuda()
    # torch.distributed.all_reduce(buf_create_cache)

    if buf_create_cache.item() != 0:
        if len(data_path) == 1:  # Single dataset.
            print(f"Creating dataset: {data_path}")
            train_dataset, eval_dataset = create_dataset(
                data_path[0], output_path,
                seed, tokenizer, end_of_conversation_token, max_seq_len)
        else:  # Blending datasets.
            train_datasets = []
            eval_datasets = []
            train_size = 0
            eval_size = 0
            for d_path in data_path:
                print(f"Creating dataset: {d_path}")
                train_dataset, eval_dataset = create_dataset(
                    d_path, output_path,
                    seed, tokenizer, end_of_conversation_token, max_seq_len)
                train_datasets.append(train_dataset)
                eval_datasets.append(eval_dataset)
                train_size += len(train_dataset)
                eval_size += len(eval_dataset)
            train_dataset = ConcatDataset(train_datasets)
            shuffle_idx = get_shuffle_idx(seed, train_size)
            train_dataset = Subset(train_dataset, shuffle_idx.tolist())
            eval_dataset = ConcatDataset(eval_datasets)
            shuffle_idx = get_shuffle_idx(seed, eval_size)
            eval_dataset = Subset(eval_dataset, shuffle_idx.tolist())

        write_diskist(train_fname, train_dataset, compression=True)
        write_diskist(eval_fname, eval_dataset, compression=True)

    # torch.distributed.barrier()
    train_dataset = InstructionDataset(tokenizer, max_seq_len, Diskist(train_fname), tokenizer.pad_token_id, 1)
    eval_dataset = InstructionDataset(tokenizer, max_seq_len, Diskist(eval_fname), tokenizer.pad_token_id, 1)
    return train_dataset, eval_dataset
