
import argparse
import re
import logging
from typing import Any
import transformers  # noqa: F401
import os
import json
from transformers import pipeline, set_seed
from transformers import AutoConfig, OPTForCausalLM, AutoModelForCausalLM, AutoTokenizer

from chatproto.conversation.history import ConversationHistory
from chatproto.registry import get_conv_settings

from katheryne.tools.chatbot import get_generator, get_model_response, get_user_input, stop_response

settings = get_conv_settings("openbuddy")
generator = get_generator("llm_trainer/lightning_logs/version_0/huggingface_format/checkpoint-step-931", settings, device="cuda:1")
set_seed(42)

instruction = """A chat between a curious user and an artificial intelligence assistant.
The assistant gives helpful, detailed, and polite answers to the user's questions."""
instruction = """Consider a conversation between User (a human) and Assistant (named Buddy).
Buddy can fluently speak the user's language (e.g. English, Chinese).
Buddy possesses vast knowledge about the world, history, and culture."""

num_rounds = 0
history = ConversationHistory(
    system=instruction,
    messages=[],
    offset=0,
    settings=settings,
)
# history.append_message(settings.roles[0], "你好")
# history.append_message(settings.roles[1], "你好呀")

while True:
    num_rounds += 1
    user_input, quit, clear = get_user_input()
    history.append_message(settings.roles[0], user_input)
    history.append_message(settings.roles[1], None)

    if quit:
        break
    elif clear:
        user_input, num_rounds = "", 0
        history.messages.clear()
        continue

    prompt = history.get_prompt()
    response = get_model_response(generator, prompt, 1024)[0]['generated_text']
    response = response[len(prompt):]
    output = stop_response(response, stop_str="\nUser: ")
    history.messages[-1][1] = output

    print("-" * 30 + f" Round {num_rounds} " + "-" * 30)
    print(f"{output}")
    # user_input = f"{output}\n\n"
