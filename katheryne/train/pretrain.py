# coding=utf-8
# Copyright 2024 XiaHan
# 
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from katheryne.train.base import train, parse_args
from katheryne.utils.hparams import HParams

def pretrain():
    args = parse_args()
    hparams = HParams.from_json_file(args.hparams)
    train_stage = hparams.get("train_stage", None)
    if train_stage is None:
        raise Exception("Please specify the train stage in the hparam file.")

    if train_stage in ["pretrain", "pretraining"]:
        from katheryne.light_modules.models.pretrain_model import PretrainLanguageModel
        from katheryne.data.loader.pretrain import create_pretrain_dataset
        train(args, hparams, create_pretrain_dataset, PretrainLanguageModel)
    else:
        raise Exception("The train stage is not consistent with the stage in config.")

if __name__ == "__main__":
    pretrain()