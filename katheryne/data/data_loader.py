# coding=utf-8
# Copyright 2024 XiaHan
# 
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import hashlib
import os
import shutil
from typing import List, Literal, Optional, Tuple, Union
import numpy as np

import datasets
import torch
from torch.utils.data import Dataset, Subset, ConcatDataset
from tqdm import tqdm

from katheryne.data.loaders.file_loader import load_dataset_from_file
from katheryne.data.loaders.hf_loader import load_dataset_from_hf
from katheryne.data.loaders.ms_loader import load_dataset_from_ms
from katheryne.data.loaders.script_loader import load_dataset_from_script
from katheryne.utils.data.data_utils import get_shuffle_idx, split_dataset
from katheryne.utils.datasets_info import DatasetInfo, DatasetPool
from katheryne.utils.hparams import HParams
from katheryne.utils.model.tokenizer_utils import load_hf_tokenizer


def create_single_dataset(hparams: HParams, dataset_info: DatasetInfo, tokenizer_path: str):
    train_stage = hparams.get("train_stage", "pretrain")
    tokenizer = load_hf_tokenizer(tokenizer_path, fast_tokenizer=True)
    
    load_from: Literal["hf_hub", "ms_hub", "script", "file"] = "hf_hub"
    if dataset_info.hf_hub_url is not None:
        load_from = "hf_hub"
    elif dataset_info.ms_hub_url is not None:
        load_from = "ms_hub"
    elif dataset_info.script_url is not None:
        load_from = "script"
    elif dataset_info.file_name is not None:
        load_from = "file"
    else:
        raise Exception("Dataset information is incomplete. Cannot determine source to load from.")
    
    if load_from == "hf_hub":
        raw_dataset = load_dataset_from_hf(hparams, dataset_info)
    elif load_from == "ms_hub":
        raw_dataset = load_dataset_from_ms(hparams, dataset_info)
    elif load_from == "script":
        raw_dataset = load_dataset_from_script(hparams, dataset_info)
    elif load_from == "file":
        raw_dataset = load_dataset_from_file(hparams, dataset_info)
    else:
        raise Exception("Invalid source to load dataset from.")
    
    # TODO:
    processed_dataset = process_dataset(raw_dataset, tokenizer)
    
    return processed_dataset

def create_datasets(hparams: HParams, datasets: DatasetPool, tokenizer_path: str):
    train_stage = hparams.get("train_stage", "pretrain")
   
    for dataset_name, dataset_info in datasets.items():
        dataset = create_single_dataset(hparams=hparams, dataset_info=dataset_info, tokenizer_path=tokenizer_path)
    
    
    data_path_obj = DatasetPath.from_data_path(data_path)

    train_datasets = []
    eval_datasets = []
    for d_path in data_path_obj:
        print(f"Creating dataset: {d_path}")
        train_dataset, eval_dataset = create_dataset(d_path, columns=["text"], preprocessor=d_path.preprocessor, seed=hparams.get("seed", 43))

        if train_dataset is not None:
            train_datasets.append(train_dataset)
        if eval_dataset is not None:
            eval_datasets.append(eval_dataset)
    
    train_dataset = datasets.concatenate_datasets(train_datasets)
    eval_dataset = datasets.concatenate_datasets(eval_datasets)

    train_dataset = PretrainDataset(train_dataset, tokenizer_path, max_seq_len, tokenizer.pad_token_id)
    eval_dataset = PretrainDataset(eval_dataset, tokenizer_path, max_seq_len, tokenizer.pad_token_id)
    return train_dataset, eval_dataset
