


import hashlib
import os
import shutil
from typing import List, Optional, Union
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

def load_plain_text(dataset_name: str, field: Union[str, List[str]], data_dir=None, data_files=None):
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

    return text_only_dataset

def create_dataset(dataset_name, output_path, seed):
    raw_dataset = load_plain_text(dataset_name, "text")

    train_dataset = raw_dataset["train"]
    eval_dataset = raw_dataset["valid"]
    return train_dataset, eval_dataset


def create_pretrain_dataset(data_path: str, output_path: str, seed: int, tokenizer, max_seq_len: int):
    """
    Creates the pretrain dataset
    """
    os.makedirs(output_path, exist_ok=True)
    data_path_list = ("_".join(data_path)).replace("/", "_").replace("\\", "_")
    tokenizer_name = tokenizer.init_kwargs["name_or_path"].replace("/", "_")
    fname = f"{data_path_list}_tokenizer{tokenizer_name}_seqlen{max_seq_len}_seed{seed}" # _tokenizer{tokenizer_name}_seqlen{max_seq_len}
    fname = "_".join(fname.split("/"))
    fname_hash = hashlib.sha256(fname.encode()).hexdigest()  # hash the file name to avoid too long file name
    train_fname = f"{output_path}/traindata_{fname_hash}"
    eval_fname = f"{output_path}/evaldata_{fname_hash}"

    cache_found = os.path.isdir(train_fname) and os.path.isdir(eval_fname)

    if not cache_found:
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
