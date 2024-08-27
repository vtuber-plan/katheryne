# coding=utf-8
# Copyright 2024 XiaHan
# 
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from katheryne.stages.base import train, parse_args
from katheryne.stages.rlhf_base import rlhf_train
from katheryne.utils.datasets_info import DatasetPool
from katheryne.utils.hparams import HParams

def auto_train_stage():
    args = parse_args()
    hparams = HParams.from_json_file(args.hparams)
    datasets_info = DatasetPool.from_json_file(args.datasets)
    train_stage = hparams.get("train_stage", None)
    if train_stage is None:
        raise Exception("Please specify the train stage in the hparam file.")
    elif train_stage in ["pretrain", "pretraining"]:
        from katheryne.light_modules.models.pretrain_model import PretrainLanguageModel
        train(args, hparams, datasets_info, PretrainLanguageModel)
    elif train_stage in ["instruction"]:
        from katheryne.light_modules.models.instruction_model import InstructionLanguageModel
        train(args, hparams, datasets_info, InstructionLanguageModel)
    elif train_stage in ["sft", "chat"]:
        from katheryne.light_modules.models.chat_model import ChatLanguageModel
        train(args, hparams, datasets_info, ChatLanguageModel)
    elif train_stage in ["reward", "rm"]:
        from katheryne.light_modules.models.reward_model import RewardLanguageModel
        train(args, hparams, datasets_info, RewardLanguageModel)
    elif train_stage in ["reward_seq", "rm_seq"]:
        from katheryne.light_modules.models.reward_model import RewardLanguageModel
        train(args, hparams, datasets_info, RewardLanguageModel)
    elif train_stage in ["ppo"]:
        from trl import PPOTrainer, PPOConfig
        rlhf_train(args, hparams, datasets_info, PPOConfig, PPOTrainer)
    elif train_stage in ["dpo"]:
        from trl import DPOTrainer, DPOConfig
        rlhf_train(args, hparams, datasets_info, DPOConfig, DPOTrainer)
    elif train_stage in ["kto"]:
        from trl import KTOTrainer, KTOConfig
        rlhf_train(args, hparams, datasets_info, KTOConfig, KTOTrainer)
    else:
        raise NotImplementedError("The train stage has not been implemented.")

if __name__ == "__main__":
    auto_train_stage()
