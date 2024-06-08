# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

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

class ChatPipeline(object):
    def __init__(self, model, tokenizer, device) -> None:
        self.model = model
        self.model.to(device)
        self.tokenizer = tokenizer
        self.device = device

    def __call__(self, input_text: str, max_new_tokens: int=256, skip_special_tokens=False) -> Any:
        input_ids = self.tokenizer([input_text], return_tensors="pt", padding='longest', max_length=4096*2, truncation=True)["input_ids"].to(self.device)
        outputs_ids = self.model.generate(inputs=input_ids, max_new_tokens=max_new_tokens,
                                          eos_token_id=self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.eos_token_id)
        outputs = self.tokenizer.batch_decode(outputs_ids, skip_special_tokens=skip_special_tokens)
        output_text = [{"generated_text": each_text} for each_text in outputs]
        return output_text

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="Directory containing trained actor model")
    parser.add_argument("--conv", type=str, default="openbuddy", help="Conversation format")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Maximum new tokens to generate per response",)
    args = parser.parse_args()
    return args

def load_local_tokenizer(path: str):
    # Locally tokenizer loading has some issue, so we need to force download
    model_json = os.path.join(path, "config.json")
    adapter_model_json = os.path.join(path, "adapter_config.json")
    if os.path.exists(model_json):
        model_json_file = json.load(open(model_json))
        model_name = model_json_file["_name_or_path"]
        if os.path.exists(model_name):
            tokenizer = AutoTokenizer.from_pretrained(model_name, fast_tokenizer=True)
        else:
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name, fast_tokenizer=True)
            except:
                tokenizer = AutoTokenizer.from_pretrained(path, fast_tokenizer=True)
    elif os.path.exists(adapter_model_json):
        model_json_file = json.load(open(adapter_model_json))
        model_name = model_json_file["base_model_name_or_path"]
        tokenizer = AutoTokenizer.from_pretrained(model_name, fast_tokenizer=True)
    return tokenizer

def load_remote_tokenizer(path: str):
    tokenizer = AutoTokenizer.from_pretrained(path, fast_tokenizer=True)
    return tokenizer

def load_local_model(path: str):
    model_json = os.path.join(path, "config.json")
    adapter_model_json = os.path.join(path, "adapter_config.json")
    if os.path.exists(adapter_model_json):
        from peft import PeftModelForCausalLM
        model_json_file = json.load(open(adapter_model_json))
        base_model_name = model_json_file["base_model_name_or_path"]
        model_config = AutoConfig.from_pretrained(base_model_name, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name, from_tf=bool(".ckpt" in path), config=model_config, trust_remote_code=True)
        
        # base_model.load_adapter(path)
        # base_model.enable_adapters()
        # model = base_model
        model = PeftModelForCausalLM.from_pretrained(base_model, path, is_trainable=True)
        model.eval()
    else:
        model_config = AutoConfig.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(path, from_tf=bool(".ckpt" in path), config=model_config, trust_remote_code=True)
    return model

def load_remote_model(path: str):
    model_config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(path, from_tf=bool(".ckpt" in path), config=model_config, trust_remote_code=True)
    return model

def get_generator(path, settings, device="cuda"):
    if os.path.exists(path):
        tokenizer = load_local_tokenizer(path)
    else:
        tokenizer = load_remote_tokenizer(path)

    tokenizer.pad_token = tokenizer.eos_token

    if os.path.exists(path):
        model = load_local_model(path)
    else:
        model = load_remote_model(path)

    # model.config.end_token_id = tokenizer.eos_token_id
    # model.config.pad_token_id = model.config.eos_token_id
    # model.resize_token_embeddings(len(tokenizer))
    # generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device="cuda", eos_token_id=tokenizer.eos_token_id)
    generator = ChatPipeline(model=model, tokenizer=tokenizer, device=device)
    return generator


def get_user_input():
    tmp = input("Enter input (type 'quit' to exit, 'clear' to clean memory): ")
    return tmp, tmp == "quit", tmp == "clear"


def get_model_response(generator, user_input, max_new_tokens):
    response = generator(user_input, max_new_tokens=max_new_tokens)
    return response


def process_response(response, num_rounds):
    output = str(response[0]["generated_text"])
    output = output.replace("<|endoftext|></s>", "")
    all_positions = [m.start() for m in re.finditer("### ", output, re.DOTALL)]
    place_of_second_q = -1
    if len(all_positions) > num_rounds*2:
        place_of_second_q = all_positions[num_rounds*2]
    if place_of_second_q != -1:
        output = output[0:place_of_second_q]
    return output

def stop_response(response: str, stop_str="<|im_end|>"):
    all_positions = [m.start() for m in re.finditer(stop_str, response, re.DOTALL)]
    if len(all_positions) < 1:
        return response
    else:
        return response[:all_positions[0]]

def main(args):
    settings = get_conv_settings(args.conv)
    generator = get_generator(args.path, settings)
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
        response = get_model_response(generator, prompt, args.max_new_tokens)[0]['generated_text']
        response = response[len(prompt):]
        output = stop_response(response)
        history.messages[-1][1] = output

        print("-" * 30 + f" Round {num_rounds} " + "-" * 30)
        print(f"{output}")
        # user_input = f"{output}\n\n"


if __name__ == "__main__":
    # Silence warnings about `max_new_tokens` and `max_length` being set
    logging.getLogger("transformers").setLevel(logging.ERROR)

    args = parse_args()
    main(args)
