# coding=utf-8
# Copyright 2024 XiaHan
# 
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import hashlib
import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

import datasets
from katheryne.data.loader import DatasetPath, create_dataset
from katheryne.datasets.reward_dataset import RewardDataset

from katheryne.utils.data.data_utils import get_shuffle_idx, split_dataset
from katheryne.utils.hparams import HParams
from katheryne.utils.model.tokenizer_utils import load_hf_tokenizer

def create_reward_dataset(hparams: HParams, data_path: List[Union[str, DatasetPath]], tokenizer_path: str, max_seq_len: int):
    """
    Creates the reward model dataset
    """
    tokenizer = load_hf_tokenizer(tokenizer_path, fast_tokenizer=True)
    data_path_obj = DatasetPath.from_data_path(data_path)
    conv_format = hparams.get("conv_format", "openbuddy")

    REWARD_FEATURES = datasets.Features({
        "history": [{
            'role': datasets.Value(dtype='string', id=None),
            'content': datasets.Value(dtype='string', id=None)
        }],
        "prompt": datasets.Value(dtype='string', id=None),
        "chosen": datasets.Value(dtype='string', id=None),
        "rejected": datasets.Value(dtype='string', id=None),
    })

    train_datasets = []
    eval_datasets = []
    for di, d_path in enumerate(data_path_obj):
        print(f"Creating dataset: {d_path}")
        train_dataset, eval_dataset = create_dataset(d_path, columns=["prompt", "chosen", "rejected"], preprocessor=d_path.preprocessor, seed=hparams.get("seed", 43))
        # train_dataset = train_dataset.cast(REWARD_FEATURES)
        # eval_dataset = eval_dataset.cast(REWARD_FEATURES)

        if train_dataset is not None:
            train_datasets.append(train_dataset)
        if eval_dataset is not None:
            eval_datasets.append(eval_dataset)
    
    train_dataset = datasets.concatenate_datasets(train_datasets)
    eval_dataset = datasets.concatenate_datasets(eval_datasets)

    train_dataset = RewardDataset(
        train_dataset,
        tokenizer_path, 
        max_seq_len,
        tokenizer.pad_token_id, 
        conv_format=conv_format, 
        end_of_conversation=hparams.get("end_of_conversation", None)
    )
    eval_dataset = RewardDataset(
        eval_dataset,
        tokenizer_path,
        max_seq_len,
        tokenizer.pad_token_id, 
        conv_format=conv_format, 
        end_of_conversation=hparams.get("end_of_conversation", None)
    )
    return train_dataset, eval_dataset
