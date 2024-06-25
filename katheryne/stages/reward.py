# coding=utf-8
# Copyright 2024 XiaHan
# 
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from katheryne.stages.base import train, parse_args
from katheryne.utils.hparams import HParams

def reward():
    args = parse_args()
    hparams = HParams.from_json_file(args.hparams)
    train_stage = hparams.get("train_stage", None)
    if train_stage is None:
        raise Exception("Please specify the train stage in the hparam file.")

    if train_stage in ["reward", "rm"]:
        from katheryne.light_modules.models.reward_model import RewardLanguageModel
        from katheryne.data.loader.reward import create_reward_dataset
        train(args, hparams, create_reward_dataset, RewardLanguageModel)
    else:
        raise Exception("The train stage is not consistent with the stage in config.")

if __name__ == "__main__":
    reward()