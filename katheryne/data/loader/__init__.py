# coding=utf-8
# Copyright 2024 XiaHan
# 
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from typing import Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field

from katheryne.utils.data.data_utils import split_dataset
from katheryne.data.preprocessors.hub import get_map_fn

class DatasetPath(BaseModel):
    path: str
    sample: Union[int, float] = 1.0
    shuffle: bool = False
    preprocessor: Optional[Union[str, List[str]]] = None

    def __str__(self) -> str:
        return self.path
    
    @classmethod
    def from_data_path(cls, data_path: List[Union[str, "DatasetPath"]]) -> List["DatasetPath"]:
        data_path_obj: List[DatasetPath] = []
        for d_path in data_path:
            if isinstance(d_path, str):
                d_path_obj = DatasetPath.model_validate({
                    "path": d_path,
                    "sample": 1.0,
                    "shuffle": False,
                    "preprocessor": None,
                })
            elif isinstance(d_path, dict):
                d_path_obj = DatasetPath.model_validate(d_path)
            else:
                raise TypeError("Invalid dataset path object, need str or dict.")
            data_path_obj.append(d_path_obj)
        return data_path_obj

import datasets

def restructure_datasets(dataset: DatasetPath, field: Union[str, List[str]], field_map: Dict[str, str]={}, split:str="train", data_dir=None, data_files=None):
    dataset_name = dataset.path
    raw_datasets = datasets.load_dataset(dataset_name, split=split, data_dir=data_dir, data_files=data_files)
    train_dataset = raw_datasets
    cols = train_dataset.column_names

    if isinstance(field, str):
        cols.remove(field)
    else:
        for each_field in field:
            cols.remove(each_field)
    train_dataset = train_dataset.remove_columns(cols)

    for old_name, new_name in field_map.items():
        if old_name in train_dataset.column_names:
            train_dataset = train_dataset.rename_column(old_name, new_name)

    return train_dataset

def create_dataset(dataset: DatasetPath, columns: List[str], preprocessor: Optional[Union[str, List[str]]]=None, seed=43) -> Tuple[datasets.Dataset, datasets.Dataset]:
    dataset_name = dataset.path
    raw_datasets = datasets.load_dataset(dataset_name)
    if "train" in raw_datasets:
        raw_train_dataset = restructure_datasets(dataset, field=columns, split="train")
    else:
        raw_train_dataset = None

    if "validation" in raw_datasets:
        raw_validation_dataset = restructure_datasets(dataset, field=columns, split="validation")
    elif "valid" in raw_datasets:
        raw_validation_dataset = restructure_datasets(dataset, field=columns, split="valid")
    elif "eval" in raw_datasets:
        raw_validation_dataset = restructure_datasets(dataset, field=columns, split="eval")
    elif "evaluation" in raw_datasets:
        raw_validation_dataset = restructure_datasets(dataset, field=columns, split="evaluation")
    else:
        raw_validation_dataset = None

    if raw_validation_dataset is None:
        train_test_valid_dataset = split_dataset(raw_train_dataset)
        train_dataset = train_test_valid_dataset["train"]
        eval_dataset = train_test_valid_dataset["valid"]
    else:
        train_dataset = raw_train_dataset
        eval_dataset = raw_validation_dataset
    
    if preprocessor is not None:
        preprocessor_fns = []
        if isinstance(preprocessor, str):
            preprocessor_fns.append(preprocessor)
        elif isinstance(preprocessor, list):
            preprocessor_fns.extend(preprocessor)
        else:
            raise Exception(f"Invalid preprocessor type {type(preprocessor)}")

        for fn_name in preprocessor_fns:
            fn = get_map_fn(fn_name)
            train_dataset = train_dataset.map(fn)
            eval_dataset = eval_dataset.map(fn)
    
    if dataset.shuffle:
        train_dataset = train_dataset.shuffle(seed=seed)

    print(f"{dataset} - Dataset size: {len(train_dataset)}")

    if isinstance(dataset.sample, int):
        sample_size = dataset.sample
        train_dataset = train_dataset.select(list(range(sample_size)))
    elif isinstance(dataset.sample, float):
        if dataset.sample != 1.0:
            sample_size = int(dataset.sample * len(train_dataset))
            train_dataset = train_dataset.select(list(range(sample_size)))
        else:
            sample_size = len(train_dataset)
    else:
        raise TypeError("Invalid sample number of dataset path object, need int or float.")

    print(f"{dataset} - Selected size: {sample_size}")

    return train_dataset, eval_dataset
