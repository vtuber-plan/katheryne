# coding=utf-8
# Copyright 2024 XiaHan
# 
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

__all__ = [
    "get_map_fn",
]

from .pretrain.merge_fields import fn_merge_fields
from .reward.rm_static import fn_rm_static


DATASET_PREPROCESSOR_FUNCTIONS = {
    "rm_static": fn_rm_static,
    "merge_fields": fn_merge_fields,
}


def get_map_fn(name: str):
    return DATASET_PREPROCESSOR_FUNCTIONS.get(name, None)