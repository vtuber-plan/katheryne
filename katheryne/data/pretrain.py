


import hashlib
import os
import shutil
from typing import Optional
import numpy as np

import datasets
import torch
from torch.utils.data import Dataset, Subset, ConcatDataset
from tqdm import tqdm
# from katheryne.data.datasets.pretrain_datasets import get_raw_dataset
# from katheryne.data.datasets import PretrainDataset, PretrainUniformDataset

from katheryne.utils.data.data_utils import get_shuffle_idx
from katheryne.utils.diskist import Diskist, extend_diskist, write_diskist
from katheryne.utils.utils import chunked

from datasets import load_dataset

def split_dataset(dataset):
    # 90% train, 10% test + validation
    train_testvalid = dataset.train_test_split(test_size=0.1)
    # Split the 10% test + valid in half test, half valid
    test_valid = train_testvalid['test'].train_test_split(test_size=0.5)
    # gather everyone if you want to have a single DatasetDict
    train_test_valid_dataset = datasets.DatasetDict({
        'train': train_testvalid['train'],
        'test': test_valid['test'],
        'valid': test_valid['train']
    })

    return train_test_valid_dataset

def load_plain_text(dataset_name, field, data_dir=None, data_files=None):
    raw_datasets = load_dataset(dataset_name, data_dir=data_dir, data_files=data_files)
    train_dataset = raw_datasets["train"]
    cols = train_dataset.column_names
    if isinstance(field, str):
        def keep_field_only(sample):
            return {"text": sample[field]}
        cols.remove(field)
        train_dataset = train_dataset.remove_columns(cols)
        if field == "text":
            text_only_dataset = train_dataset
        else:
            text_only_dataset = train_dataset.rename_column(field, "text")
    else:
        def keep_field_only(sample):
            out = []
            for subfield in field:
                data = sample[subfield]
                if isinstance(data, str):
                    line = data
                elif isinstance(data, list):
                    line = "\n".join(data)
                else:
                    line = str(data)
                out.append(line)
            return {"text": "\n".join(out)}
        for subfield in field:
            cols.remove(subfield)
        train_dataset = train_dataset.remove_columns(cols)
        text_only_dataset = train_dataset.map(keep_field_only, num_proc=8)

    return split_dataset(text_only_dataset)

def get_raw_dataset(dataset_name, seed):
    return load_plain_text(dataset_name, "text")

def create_dataset(dataset_name, output_path, seed):
    raw_dataset = get_raw_dataset(dataset_name, seed)

    train_dataset = raw_dataset["train"]
    eval_dataset = raw_dataset["valid"]
    return train_dataset, eval_dataset


def create_pretrain_dataset(data_path, output_path, seed, tokenizer, max_seq_len):
    """
    Creates the pretrain dataset
    """
    os.makedirs(output_path, exist_ok=True)
    fname = "_".join(data_path)
    tokenizer_name = tokenizer.init_kwargs["name_or_path"].replace("/", "_")
    fname = f"{fname}_seed{seed}" # _tokenizer{tokenizer_name}_seqlen{max_seq_len}
    fname = "_".join(fname.split("/"))
    fname = hashlib.sha256(fname.encode()).hexdigest()  # hash the file name to avoid too long file name
    train_fname = f"{output_path}/traindata_{fname}"
    eval_fname = f"{output_path}/evaldata_{fname}"

    cache_found = os.path.isdir(train_fname) and os.path.isdir(eval_fname)
    buf_create_cache = torch.ByteTensor([not cache_found]).cuda()
    # torch.distributed.all_reduce(buf_create_cache)

    if buf_create_cache.item() != 0:
        if len(data_path) == 1:  # Single dataset.
            print(f"Creating dataset: {data_path}")
            train_dataset, eval_dataset = create_dataset(data_path[0], output_path, seed)
        else:  # Blending datasets.
            train_datasets = []
            eval_datasets = []
            for d_path in data_path:
                print(f"Creating dataset: {d_path}")
                train_dataset, eval_dataset = create_dataset(d_path, output_path, seed)
                train_datasets.append(train_dataset)
                eval_datasets.append(eval_dataset)
            train_dataset = datasets.concatenate_datasets(train_datasets)
            eval_dataset = datasets.concatenate_datasets(eval_datasets)
        
        # train_dataset.save_to_disk(train_fname, max_shard_size="4GB", num_proc=8)
        # eval_dataset.save_to_disk(eval_fname, max_shard_size="4GB", num_proc=8)
    # train_dataset = datasets.load_from_disk(train_fname)
    # eval_dataset = datasets.load_from_disk(eval_fname)

    # torch.distributed.barrier()
    train_dataset = PretrainDataset(tokenizer, max_seq_len, train_dataset, tokenizer.pad_token_id)
    eval_dataset = PretrainDataset(tokenizer, max_seq_len, eval_dataset, tokenizer.pad_token_id)
    return train_dataset, eval_dataset
