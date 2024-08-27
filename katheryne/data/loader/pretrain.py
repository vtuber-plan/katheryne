# coding=utf-8
# Copyright 2024 XiaHan
# 
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import hashlib
import os
import shutil
from typing import List, Optional, Tuple, Union
import numpy as np

import datasets
import torch
from torch.utils.data import Dataset, Subset, ConcatDataset
from tqdm import tqdm
from katheryne.data.loader import DatasetPath, create_dataset
from katheryne.datasets.pretrain_dataset import PretrainDataset
# from katheryne.data.datasets import PretrainDataset, PretrainUniformDataset

from katheryne.utils.data.data_utils import get_shuffle_idx, split_dataset
from katheryne.utils.datasets_info import DatasetPool
from katheryne.utils.diskist import Diskist, extend_diskist, write_diskist
from katheryne.utils.hparams import HParams
from katheryne.utils.model.tokenizer_utils import load_hf_tokenizer
from katheryne.utils.utils import chunked

from datasets import load_dataset

def create_pretrain_dataset(hparams: HParams, datasets: DatasetPool, tokenizer_path: str, max_seq_len: int):
    """
    Creates the pretrain dataset
    """
    tokenizer = load_hf_tokenizer(tokenizer_path, fast_tokenizer=True)
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
