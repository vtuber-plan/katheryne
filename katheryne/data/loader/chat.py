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
from katheryne.datasets.chat_dataset import ChatDataset
from katheryne.datasets.pretrain_dataset import PretrainDataset

from katheryne.utils.data.data_utils import get_shuffle_idx, split_dataset
from katheryne.utils.diskist import Diskist, extend_diskist, write_diskist
from katheryne.utils.hparams import HParams
from katheryne.utils.model.tokenizer_utils import load_hf_tokenizer
from katheryne.utils.utils import chunked

def create_chat_dataset(hparams: HParams, data_path: List[Union[str, DatasetPath]], tokenizer_path: str, max_seq_len: int):
    """
    Creates the chat dataset
    """
    tokenizer = load_hf_tokenizer(tokenizer_path, fast_tokenizer=True)
    data_path_obj = DatasetPath.from_data_path(data_path)

    conv_format = hparams.get("conv_format", "openbuddy")

    CHAT_FEATURES = datasets.Features({'messages': [{
        'role': datasets.Value(dtype='string', id=None),
        'content': datasets.Value(dtype='string', id=None)
        }]
    })

    train_datasets = []
    eval_datasets = []
    for di, d_path in enumerate(data_path_obj):
        print(f"Creating dataset: {d_path}")
        train_dataset, eval_dataset = create_dataset(d_path, columns=["messages"], preprocessor=d_path.preprocessor, seed=hparams.get("seed", 43))
        train_dataset = train_dataset.cast(CHAT_FEATURES)
        eval_dataset = eval_dataset.cast(CHAT_FEATURES)

        if train_dataset is not None:
            train_datasets.append(train_dataset)
        if eval_dataset is not None:
            eval_datasets.append(eval_dataset)
    
    train_dataset = datasets.concatenate_datasets(train_datasets)
    eval_dataset = datasets.concatenate_datasets(eval_datasets)

    train_dataset = ChatDataset(
        train_dataset,
        tokenizer_path, 
        max_seq_len,
        tokenizer.pad_token_id, 
        conv_format=conv_format, 
        end_of_conversation=hparams.get("end_of_conversation", None)
    )
    eval_dataset = ChatDataset(
        eval_dataset,
        tokenizer_path,
        max_seq_len,
        tokenizer.pad_token_id, 
        conv_format=conv_format, 
        end_of_conversation=hparams.get("end_of_conversation", None)
    )
    return train_dataset, eval_dataset
