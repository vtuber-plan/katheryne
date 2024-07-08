# coding=utf-8
# Copyright 2024 XiaHan
# 
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from katheryne.stages.base import parse_args
from katheryne.stages.rlhf_base import rlhf_train
from katheryne.utils.hparams import HParams

def dpo():
    args = parse_args()
    hparams = HParams.from_json_file(args.hparams)
    train_stage = hparams.get("train_stage", None)
    if train_stage is None:
        raise Exception("Please specify the train stage in the hparam file.")

    if train_stage in ["dpo"]:
        from katheryne.data.loader.rlhf import create_rlhf_dataset
        from trl import DPOTrainer, DPOConfig
        rlhf_train(args, hparams, create_rlhf_dataset, DPOConfig, DPOTrainer)
    else:
        raise Exception("The train stage is not consistent with the stage in config.")

if __name__ == "__main__":
    dpo()
