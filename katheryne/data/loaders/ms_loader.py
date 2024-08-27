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

def load_dataset_from_ms(hparams: HParams, dataset_info: DatasetInfo) -> Union[
    datasets.DatasetDict,
    datasets.Dataset,
    datasets.IterableDatasetDict,
    datasets.IterableDataset,
]:
    data_path = dataset_info.ms_hub_url
    data_name = dataset_info.subset
    data_dir = dataset_info.folder
    data_files = None

    from modelscope import MsDataset
    from modelscope.utils.config_ds import MS_DATASETS_CACHE

    cache_dir = hparams.get("cache_dir", None) or MS_DATASETS_CACHE
    dataset = MsDataset.load(
        dataset_name=data_path,
        subset_name=data_name,
        data_dir=data_dir,
        data_files=data_files,
        split=dataset_info.split,
        cache_dir=cache_dir,
        token=hparams.get("ms_hub_token", None),
        use_streaming=hparams.get("streaming", None),
    )
    if isinstance(dataset, MsDataset):
        dataset = dataset.to_hf_dataset()
    return dataset
