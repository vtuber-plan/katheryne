
import os
import json
from typing import List
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerBase

def load_hf_tokenizer(model_name_or_path, fast_tokenizer=True, trust_remote_code=True, padding_side="left"):
    if "open_llama" in model_name_or_path:
        fast_tokenizer = False
    if os.path.exists(model_name_or_path):
        # Locally tokenizer loading has some issue, so we need to force download
        model_json = os.path.join(model_name_or_path, "config.json")
        if os.path.exists(model_json):
            model_json_file = json.load(open(model_json))
            if "_name_or_path" in model_json_file:
                model_name = model_json_file["_name_or_path"]
            else:
                model_name = model_name_or_path
            if os.path.exists(model_name):
                tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=fast_tokenizer, trust_remote_code=trust_remote_code, padding_side=padding_side)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=fast_tokenizer, trust_remote_code=trust_remote_code, padding_side=padding_side)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=fast_tokenizer, trust_remote_code=trust_remote_code, padding_side=padding_side)
    
    print(f"Tokenizer {model_name_or_path} is_fast: ", tokenizer.is_fast)
    if tokenizer._pad_token is None:
        try:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"Tokenizer {model_name_or_path} pad_token missing, use eos_token instead.")
        except AttributeError as e:
            print(f"Failed to set the eos_token of tokenizer {model_name_or_path}")
    return tokenizer

def pad_tokenizer(tokenizer: PreTrainedTokenizer, pad_to: int=64) -> PreTrainedTokenizer:
    current_vocab_len = len(tokenizer.get_vocab())
    if current_vocab_len % pad_to == 0:
        return tokenizer
    padded_vocab_len = ((current_vocab_len // pad_to) + 1) * pad_to
    add_token_num = padded_vocab_len - current_vocab_len
    tokenizer.add_tokens([f"TOKENIZER_PAD_TOKEN_{i}" for i in range(add_token_num)])

def get_text_offset(tokenizer: PreTrainedTokenizerBase, text: str, tokens: List[str]):
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

def is_merge_prefix_space(tokenizer: PreTrainedTokenizerBase) -> bool:
    lhs = tokenizer(": a", add_special_tokens=False)['input_ids']
    rhs = tokenizer(": ", add_special_tokens=False)['input_ids']
    return len(lhs) == len(rhs)