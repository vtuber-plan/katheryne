
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
instruction = ""

num_rounds = 0
history = ConversationHistory(
    system=instruction,
    messages=[],
    offset=0,
    settings=settings,
)
with open("a.java", "r", encoding="utf-8") as f:
    user_input = f.read()

history.append_message(settings.roles[0], "Please translate the following code into cangjie:\n```\n" + user_input + "\n```")
history.append_message(settings.roles[1], None)

prompt = history.get_prompt()
response = get_model_response(generator, prompt, 2048)[0]['generated_text']
response = response[len(prompt):]
output = response
# output = stop_response(response)
history.messages[-1][1] = output

print("-" * 30 + f" Round {num_rounds} " + "-" * 30)
print(f"{output}")
# user_input = f"{output}\n\n"
