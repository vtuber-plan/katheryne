# coding=utf-8
# Copyright 2024 XiaHan
# 
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from katheryne.stages.base import train, parse_args
from katheryne.stages.rlhf_base import rlhf_train
from katheryne.utils.hparams import HParams

def auto_train_stage():
    args = parse_args()
    hparams = HParams.from_json_file(args.hparams)
    train_stage = hparams.get("train_stage", None)
    if train_stage is None:
        raise Exception("Please specify the train stage in the hparam file.")
    elif train_stage in ["pretrain", "pretraining"]:
        from katheryne.light_modules.models.pretrain_model import PretrainLanguageModel
        from katheryne.data.loader.pretrain import create_pretrain_dataset
        train(args, hparams, create_pretrain_dataset, PretrainLanguageModel)
    elif train_stage in ["instruction"]:
        from katheryne.light_modules.models.instruction_model import InstructionLanguageModel
        from katheryne.data.loader.instruction import create_instruction_dataset
        train(args, hparams, create_instruction_dataset, InstructionLanguageModel)
    elif train_stage in ["sft", "chat"]:
        from katheryne.light_modules.models.chat_model import ChatLanguageModel
        from katheryne.data.loader.chat import create_chat_dataset
        train(args, hparams, create_chat_dataset, ChatLanguageModel)
    elif train_stage in ["reward", "rm"]:
        from katheryne.light_modules.models.reward_model import RewardLanguageModel
        from katheryne.data.loader.reward import create_reward_dataset
        train(args, hparams, create_reward_dataset, RewardLanguageModel)
    elif train_stage in ["reward_seq", "rm_seq"]:
        from katheryne.light_modules.models.reward_model import RewardLanguageModel
        from katheryne.data.loader.reward import create_reward_dataset
        train(args, hparams, create_reward_dataset, RewardLanguageModel)
    elif train_stage in ["ppo"]:
        from katheryne.data.loader.rlhf import create_rlhf_dataset
        rlhf_train(args, hparams, create_rlhf_dataset)
    else:
        raise NotImplementedError("The train stage has not been implemented.")

if __name__ == "__main__":
    auto_train_stage()
