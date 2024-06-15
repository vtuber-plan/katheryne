


import hashlib
import os
import shutil
from typing import List, Optional, Tuple, Union
import numpy as np

import datasets
import torch
from torch.utils.data import Dataset, Subset, ConcatDataset
from tqdm import tqdm
from katheryne.data.loader import DatasetPath
from katheryne.datasets.pretrain_dataset import PretrainDataset
# from katheryne.data.datasets.pretrain_datasets import get_raw_dataset
# from katheryne.data.datasets import PretrainDataset, PretrainUniformDataset

from katheryne.utils.data.data_utils import get_shuffle_idx, split_dataset
from katheryne.utils.diskist import Diskist, extend_diskist, write_diskist
from katheryne.utils.hparams import HParams
from katheryne.utils.model.tokenizer_utils import load_hf_tokenizer
from katheryne.utils.utils import chunked

from datasets import load_dataset

def load_plain_text(dataset_name: str, field: Union[str, List[str]], split:str="train", data_dir=None, data_files=None):
    raw_datasets = load_dataset(dataset_name, split=split, data_dir=data_dir, data_files=data_files)
    train_dataset = raw_datasets
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
    raw_datasets = load_dataset(dataset_name)
    if "train" in raw_datasets:
        raw_train_dataset = load_plain_text(dataset_name, "text", split="train")
    else:
        raw_train_dataset = None
    
    if "validation" in raw_datasets:
        raw_validation_dataset = load_plain_text(dataset_name, "text", split="validation")
    elif "valid" in raw_datasets:
        raw_validation_dataset = load_plain_text(dataset_name, "text", split="valid")
    elif "eval" in raw_datasets:
        raw_validation_dataset = load_plain_text(dataset_name, "text", split="eval")
    elif "evaluation" in raw_datasets:
        raw_validation_dataset = load_plain_text(dataset_name, "text", split="evaluation")
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


def create_pretrain_dataset(hparams: HParams, data_path: List[Union[str, DatasetPath]], output_path: str, seed: int, tokenizer_path: str, max_seq_len: int):
    """
    Creates the pretrain dataset
    """
    tokenizer = load_hf_tokenizer(tokenizer_path, fast_tokenizer=True)
    data_path_obj = []
    for d_path in data_path:
        if isinstance(d_path, str):
            d_path_obj = DatasetPath.model_validate({
                "path": d_path,
                "sample": 1.0,
                "shuffle": False
            })
        elif isinstance(d_path, dict):
            d_path_obj = DatasetPath.model_validate(d_path)
        else:
            raise TypeError("Invalid dataset path object, need str or dict.")
        data_path_obj.append(d_path_obj)
    
    os.makedirs(output_path, exist_ok=True)
    data_path_list = ("_".join([str(p) for p in data_path_obj])).replace("/", "_").replace("\\", "_")
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
        for d_path in data_path_obj:
            print(f"Creating dataset: {d_path}")
            train_dataset, eval_dataset = create_dataset(d_path.path, output_path, seed)

            if d_path.shuffle:
                train_dataset = train_dataset.shuffle(seed=hparams.get("seed", 43))

            if isinstance(d_path.sample, int):
                sample_size = d_path.sample
                train_dataset = train_dataset.select(list(range(sample_size)))
            elif isinstance(d_path.sample, float):
                if d_path.sample != 1.0:
                    sample_size = int(d_path.sample * len(train_dataset))
                    train_dataset = train_dataset.select(list(range(sample_size)))
            else:
                raise TypeError("Invalid sample number of dataset path object, need int or float.")
    
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
    train_dataset = PretrainDataset(tokenizer_path, max_seq_len, train_dataset, tokenizer.pad_token_id)
    eval_dataset = PretrainDataset(tokenizer_path, max_seq_len, eval_dataset, tokenizer.pad_token_id)
    return train_dataset, eval_dataset
