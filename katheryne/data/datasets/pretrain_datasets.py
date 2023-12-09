from typing import List, Union
import torch
from torch.utils.data import Dataset, Subset, ConcatDataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import datasets
from datasets import load_dataset
import numpy as np
import os
import hashlib

from tqdm import tqdm
from chatproto.conversation.history import ConversationHistory
from chatproto.registry import get_conv_settings

from katheryne.utils.data.data_utils import get_shuffle_idx, get_subset


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

def load_camelai_math_text(dataset_name):
    raw_datasets = load_dataset(dataset_name)
    train_dataset = raw_datasets["train"]

    settings = get_conv_settings("ningyu")
    def keep_field_only(sample):
        sub_topic = sample["sub_topic"]
        message_1 = sample["message_1"]
        message_2 = sample["message_2"]

        history = ConversationHistory(
            system=sub_topic,
            messages=[("user", message_1), ("assistant", message_2)],
            offset=0,
            settings=settings,
        )

        return {"text": history.get_prompt()}
    text_only_dataset = train_dataset.map(keep_field_only, num_proc=8)

    return split_dataset(text_only_dataset)

def load_star_coder_data(dataset_name):
    def keep_field_only(sample):
        return {"text": sample["content"]}
    raw_datasets = []
    for subdataset_name in [
            "python", "tex", "yaml", "php", "makefile", "mathematica", "markdown",
            "java", "javascript", "html", "go", "json", "typescript", "matlab", "css", "dockerfile",
            "julia", "lua", "ocaml", "kotlin", "r", "c", "cpp", "batchfile", "cmake", "c-sharp", "haskell", "fortran", "cuda"
        ]:
        subdataset = load_dataset(dataset_name, data_dir=subdataset_name, split="train")

        cols = subdataset.column_names
        cols.remove("content")
        subdataset = subdataset.remove_columns(cols)
        subdataset = subdataset.rename_column("content", "text")

        raw_datasets.append(subdataset)
    
    raw_dataset = datasets.concatenate_datasets(raw_datasets)

    return split_dataset(raw_dataset)

def get_raw_dataset(dataset_name, seed):
    if "pile" in dataset_name:
        return load_plain_text(dataset_name, "text")
    elif "starcoderdata" in dataset_name:
        return load_star_coder_data(dataset_name)
    elif "WuDaoCorpus2" in dataset_name:
        return load_plain_text(dataset_name, ["title", "content"])
    elif "roots_zh_wikibooks" in dataset_name:
        return load_plain_text(dataset_name, "text", data_dir="data")
    elif "roots_zh-cn_wikipedia" in dataset_name:
        return load_plain_text(dataset_name, "text", data_dir="data")
    elif "roots_zh_wikinews" in dataset_name:
        return load_plain_text(dataset_name, "text", data_dir="data")
    elif "chinese_poetrys" in dataset_name:
        return load_plain_text(dataset_name, ["title", "author", "paragraphs"], data_files=[f"00{i}.json" for i in range(1,9)])
    else:
        return load_plain_text(dataset_name, "text")
