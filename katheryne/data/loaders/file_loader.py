# coding=utf-8
# Copyright 2024 XiaHan
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import os
from typing import Union
import datasets

from katheryne.utils.datasets_info import DatasetInfo
from katheryne.utils.hparams import HParams

DATASET_FILE_EXT = ["arrow", "csv", "json", "jsonl", "parquet", "txt"]

def load_dataset_from_file(hparams: HParams, dataset_info: DatasetInfo) -> Union[
    datasets.DatasetDict,
    datasets.Dataset,
    datasets.IterableDatasetDict,
    datasets.IterableDataset,
]:
    data_path, data_name, data_dir, data_files = None, None, None, None
    
    data_files = []
    local_path = dataset_info.file_name
    if os.path.isdir(local_path):  # is directory
        for file_name in os.listdir(local_path):
            data_files.append(os.path.join(local_path, file_name))
            if data_path is None:
                data_path = file_name.split(".")[-1]
            elif data_path != file_name.split(".")[-1]:
                raise ValueError("File types should be identical.")
    elif os.path.isfile(local_path):  # is file
        data_files.append(local_path)
        data_path = local_path.split(".")[-1]
    else:
        raise ValueError("File {} not found.".format(local_path))

    if data_path is None:
        raise ValueError("Allowed file types: {}.".format(",".join(DATASET_FILE_EXT)))

    dataset = datasets.load_dataset(
        path=data_path,
        name=data_name,
        data_dir=data_dir,
        data_files=data_files,
        split=dataset_info.split,
        cache_dir=hparams.get("cache_dir", None),
        token=hparams.get("hf_hub_token", None),
        trust_remote_code=True,  # TODO
    )
    return dataset
