

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizerBase
)
from transformers.utils import PaddingStrategy

import torch
from torch.utils.data import Dataset, Subset, ConcatDataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from datasets import load_dataset
import numpy as np
import os
import hashlib
from itertools import chain

from tqdm import tqdm


@dataclass
class DataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        first_feature = features[0]
        keys = first_feature.keys()
        batch = {}
        for k in keys:
            batch_k = self.tokenizer.pad(
                [{"input_ids": feat[k]} for feat in features],
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )
            batch[k] = batch_k["input_ids"]
        return batch

@dataclass
class DataCollatorReward:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        first_feature = features[0]
        keys = first_feature.keys()

        max_length = 0
        for k in keys:
            lst = [feat[k] for feat in features]
            max_value = max([item.shape[0] for item in lst])
            max_length = max(max_length, max_value)

        if self.padding or self.padding == "longest":
            padding_strategy = "max_length"
            padding_length = min(max_length, self.max_length)
        elif self.padding == "max_length":
            padding_strategy = "max_length"
            padding_length = self.max_length
        else:
            padding_strategy = "do_not_pad"
            padding_length = self.max_length

        batch = {}
        for k in keys:
            batch_k = self.tokenizer.pad(
                [{"input_ids": feat[k]} for feat in features],
                padding=padding_strategy,
                max_length=padding_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )
            batch[k] = batch_k["input_ids"]
        return batch
