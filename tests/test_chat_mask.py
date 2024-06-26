# coding=utf-8
# Copyright 2024 XiaHan
# 
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from typing import List
import numpy as np
import torch

from chatproto.conversation.history import ConversationHistory
from chatproto.registry import get_conv_settings

from transformers.trainer_pt_utils import LabelSmoother
from katheryne.utils.model.tokenizer_utils import load_hf_tokenizer
from katheryne.utils.utils import searchsorted
IGNORE_TOKEN_ID = LabelSmoother.ignore_index

tokenizer = load_hf_tokenizer("meta-llama/Llama-2-7b-hf", fast_tokenizer=True)
end_of_conversation = tokenizer.eos_token

def get_text_offset(text: str, tokens: List[str]):
    if tokenizer.is_fast:
        text_offset = [-1] * len(tokens)
        batch_encoding = tokenizer([text])
        for token_i in range(len(tokens)):
            span = batch_encoding.token_to_chars(0, token_i)
            if span is None:
                continue
            start, end = span
            text_offset[token_i] = start
    else:
        text_offset = []
        for token_i in range(0, len(tokens)):
            if token_i == 0:
                text_offset.append(-1)
                continue
            prefix_text = tokenizer.convert_tokens_to_string(tokens[:token_i])
            if text.startswith(prefix_text):
                text_offset.append(len(prefix_text))
            else:
                text_offset.append(-1)
        
        last_id = len(text)
        for token_i in reversed(range(0, len(tokens))):
            if text_offset[token_i] == -1:
                text_offset[token_i] = last_id
            else:
                last_id = text_offset[token_i]
    return text_offset

def mask_label(prompt: str, target, indices):
    tokens = tokenizer.convert_ids_to_tokens(target, skip_special_tokens=True)
    text_offset = get_text_offset(prompt, tokens)

    cur_len = 1
    target[:cur_len] = IGNORE_TOKEN_ID
    start = cur_len
    end = cur_len
    for i, turn in enumerate(indices[1:]):
        if i % 2 == 0:
            continue
        text_start, text_end = turn
        if prompt[text_start - 1] == " ":
            text_start -= 1
        start = np.searchsorted(text_offset, text_start)
        end = np.searchsorted(text_offset, text_end)
        target[cur_len:start] = IGNORE_TOKEN_ID
        cur_len = end

    end_conv = np.searchsorted(text_offset, len(prompt)-len(end_of_conversation))
    target[end_conv:end] = IGNORE_TOKEN_ID

    if True:  # Inspect and check the correctness of masking
        z = target.clone()
        z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
        print(prompt)
        print("================")
        print(tokenizer.decode(z))
    return target

settings = get_conv_settings("openbuddy")
history = ConversationHistory(
    system="You are a robot",
    messages=[
        (settings.roles[0], "Hello, who are you?"),
        (settings.roles[1], "I am a robot."),
        (settings.roles[0], "Really?"),
        (settings.roles[1], "Yes, I am a robot"),
        (settings.roles[0], "Really??"),
        (settings.roles[1], "Yes, I am a robot.."),
    ],
    offset=0,
    settings=settings,
)

prompt, indices = history.get_prompt_and_indices()
prompt += end_of_conversation
encoded_text = tokenizer(prompt,
                max_length=512,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=True,
            )
labels = encoded_text["input_ids"].squeeze(0).clone()
mask_label(prompt, labels, indices)