# coding=utf-8
# Copyright 2024 XiaHan
# 
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

__all__ = [
    "get_map_fn",
]

from .pretrain.merge_fields import fn_merge_fields as pretrain_fn_merge_fields
from .reward.rm_static import fn_rm_static as reward_rm_static
from .rlhf.rm_static import fn_rm_static as rlhf_fn_rm_static

DATASET_PREPROCESSOR_FUNCTIONS = {
    "reward_rm_static": reward_rm_static,
    "rlhf_rm_static": rlhf_fn_rm_static,
    "pretrain_merge_fields": pretrain_fn_merge_fields,
}


def get_map_fn(name: str):
    return DATASET_PREPROCESSOR_FUNCTIONS.get(name, None)