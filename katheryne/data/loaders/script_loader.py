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

def load_dataset_from_script(hparams: HParams, dataset_info: DatasetInfo) -> Union[
    datasets.DatasetDict,
    datasets.Dataset,
    datasets.IterableDatasetDict,
    datasets.IterableDataset,
]:
    data_path = dataset_info.script_url
    data_name = dataset_info.subset
    data_dir = dataset_info.folder
    data_files = None

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
