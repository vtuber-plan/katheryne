
import os
import json
from transformers import AutoTokenizer, PreTrainedTokenizer

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
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=fast_tokenizer, trust_remote_code=trust_remote_code, padding_side=padding_side)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=fast_tokenizer, trust_remote_code=trust_remote_code, padding_side=padding_side)
    
    print(f"Tokenizer {model_name_or_path} is_fast: ", tokenizer.is_fast)
    return tokenizer

def pad_tokenizer(tokenizer: PreTrainedTokenizer, pad_to: int=64) -> PreTrainedTokenizer:
    current_vocab_len = len(tokenizer.get_vocab())
    if current_vocab_len % pad_to == 0:
        return tokenizer
    padded_vocab_len = ((current_vocab_len // pad_to) + 1) * pad_to
    add_token_num = padded_vocab_len - current_vocab_len
    tokenizer.add_tokens([f"TOKENIZER_PAD_TOKEN_{i}" for i in range(add_token_num)])
