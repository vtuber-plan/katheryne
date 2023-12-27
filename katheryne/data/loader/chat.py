
import hashlib
import os
import shutil
from typing import List, Optional, Tuple, Union
import numpy as np

import datasets
import torch
from torch.utils.data import Dataset, Subset, ConcatDataset
from tqdm import tqdm
from katheryne.datasets.chat_dataset import ChatDataset
from katheryne.datasets.pretrain_dataset import PretrainDataset

from katheryne.utils.data.data_utils import get_shuffle_idx, split_dataset
from katheryne.utils.diskist import Diskist, extend_diskist, write_diskist
from katheryne.utils.hparams import HParams
from katheryne.utils.utils import chunked

from datasets import load_dataset

def load_chat_messages(dataset_name: str, field: Union[str, List[str]], split:str="train", data_dir=None, data_files=None):
    raw_datasets = load_dataset(dataset_name, split=split, data_dir=data_dir, data_files=data_files)
    train_dataset = raw_datasets
    cols = train_dataset.column_names

    cols.remove(field)
    train_dataset = train_dataset.remove_columns(cols)
    if field == "messages":
        messages_only_dataset = train_dataset
    else:
        messages_only_dataset = train_dataset.rename_column(field, "messages")

    return messages_only_dataset

def create_dataset(dataset_name, output_path, seed) -> Tuple[datasets.Dataset, datasets.Dataset]:
    raw_datasets = load_dataset(dataset_name)
    if "train" in raw_datasets:
        raw_train_dataset = load_chat_messages(dataset_name, "messages", split="train")
    else:
        raw_train_dataset = None
    
    if "validation" in raw_datasets:
        raw_validation_dataset = load_chat_messages(dataset_name, "messages", split="validation")
    elif "valid" in raw_datasets:
        raw_validation_dataset = load_chat_messages(dataset_name, "messages", split="valid")
    elif "eval" in raw_datasets:
        raw_validation_dataset = load_chat_messages(dataset_name, "messages", split="eval")
    elif "evaluation" in raw_datasets:
        raw_validation_dataset = load_chat_messages(dataset_name, "messages", split="evaluation")
    else:
        raw_validation_dataset = None

    if raw_validation_dataset is None:
        train_test_valid_dataset = split_dataset(raw_train_dataset)
        train_dataset = train_test_valid_dataset["train"]
        eval_dataset = train_test_valid_dataset["valid"]
    else:
        train_dataset = raw_train_dataset
        eval_dataset = raw_validation_dataset
    return train_dataset, eval_dataset


def create_chat_dataset(hparams: HParams, data_path: str, output_path: str, seed: int, tokenizer, max_seq_len: int):
    """
    Creates the chat dataset
    """
    conv_format = hparams.get("conv_format", "openbuddy")
    os.makedirs(output_path, exist_ok=True)
    data_path_list = ("_".join(data_path)).replace("/", "_").replace("\\", "_")
    tokenizer_name = tokenizer.init_kwargs["name_or_path"].replace("/", "_")
    fname = f"{data_path_list}_tokenizer{tokenizer_name}_seqlen{max_seq_len}_seed{seed}" # _tokenizer{tokenizer_name}_seqlen{max_seq_len}
    fname = "_".join(fname.split("/"))
    fname_hash = hashlib.sha256(fname.encode()).hexdigest()  # hash the file name to avoid too long file name
    train_fname = f"{output_path}/traindata_{fname_hash}"
    eval_fname = f"{output_path}/evaldata_{fname_hash}"

    cache_found = os.path.isdir(train_fname) and os.path.isdir(eval_fname)

    CHAT_FEATURES = datasets.Features({'messages': [{
        'role': datasets.Value(dtype='string', id=None),
        'content': datasets.Value(dtype='string', id=None)
        }]
    })
    if not cache_found:
        train_datasets = []
        eval_datasets = []
        for di, d_path in enumerate(data_path):
            print(f"Creating dataset: {d_path}")
            train_dataset, eval_dataset = create_dataset(d_path, output_path, seed)
            train_dataset = train_dataset.cast(CHAT_FEATURES)
            eval_dataset = eval_dataset.cast(CHAT_FEATURES)

            if train_dataset is not None:
                train_datasets.append(train_dataset)
            if eval_dataset is not None:
                eval_datasets.append(eval_dataset)
        
        train_dataset = datasets.concatenate_datasets(train_datasets)
        eval_dataset = datasets.concatenate_datasets(eval_datasets)
    
        # train_dataset.save_to_disk(train_fname, max_shard_size="4GB", num_proc=8)
        # eval_dataset.save_to_disk(eval_fname, max_shard_size="4GB", num_proc=8)
    # train_dataset = datasets.load_from_disk(train_fname)
    # eval_dataset = datasets.load_from_disk(eval_fname)

    # torch.distributed.barrier()
    train_dataset = ChatDataset(tokenizer, 
        max_seq_len, 
        train_dataset, 
        tokenizer.pad_token_id, 
        conv_format=conv_format, 
        end_of_conversation=hparams.get("end_of_conversation", None)
    )
    eval_dataset = ChatDataset(tokenizer, 
        max_seq_len, 
        eval_dataset, 
        tokenizer.pad_token_id, 
        conv_format=conv_format, 
        end_of_conversation=hparams.get("end_of_conversation", None)
    )
    return train_dataset, eval_dataset
