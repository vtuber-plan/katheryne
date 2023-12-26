import numpy as np
import torch
import datasets
from torch.utils.data import Dataset

from transformers import PreTrainedTokenizerBase

from transformers.trainer_pt_utils import LabelSmoother

from chatproto.conversation.history import ConversationHistory
from chatproto.registry import get_conv_settings

from katheryne.utils.model.tokenizer_utils import get_text_offset

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

class ChatDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_seq_len: int, dataset: datasets.Dataset, pad_token_id: int, conv_format: str="openbuddy") -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.dataset = dataset
        self.pad_token_id = pad_token_id

        self.settings = get_conv_settings(conv_format)
        self.end_of_conversation = self.tokenizer.eos_token

    def __len__(self):
        return len(self.dataset)

    def tokenize(self, text: str, add_special_tokens: bool=True):
        encoded_text = self.tokenizer(text,
                        max_length=self.max_seq_len,
                        truncation=True,
                        return_tensors="pt",
                        add_special_tokens=add_special_tokens,
                    )
        return encoded_text
    
    def get_prompt(self, messages, ignore_last:bool=False):
        system = ""
        for i, item in enumerate(messages):
            role, content = item["role"], item["content"]
            if role == "system":
                system = content
                break
        history = ConversationHistory(
            system=system,
            messages=[],
            offset=0,
            settings=self.settings,
        )

        for i, item in enumerate(messages):
            role, content = item["role"], item["content"]
            if role == "system":
                continue
            if ignore_last and i == len(messages) - 1:
                content = None
            if role.lower() == "user":
                real_role = self.settings.roles[0]
            elif role.lower() == "assistant":
                real_role = self.settings.roles[1]
            else:
                raise Exception(f"Unknown role {role}")
            history.messages.append((real_role, content))
        return history.get_prompt_and_indices()

    def mask_label(self, prompt: str, target, indices):
        tokens = self.tokenizer.convert_ids_to_tokens(target, skip_special_tokens=True)
        text_offset = get_text_offset(self.tokenizer, prompt, tokens)

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

        end_conv = np.searchsorted(text_offset, len(prompt)-len(self.end_of_conversation))
        target[end_conv:end] = IGNORE_TOKEN_ID
        if False:  # Inspect and check the correctness of masking
            z = target.clone()
            z = torch.where(z == IGNORE_TOKEN_ID, self.tokenizer.unk_token_id, z)
            print(prompt)
            print("================")
            print(self.tokenizer.decode(z))
        return target

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        messages = sample["messages"]
        prompt, indices = self.get_prompt(messages)
        prompt += self.end_of_conversation

        encoded_text = self.tokenize(prompt)
        labels = encoded_text["input_ids"].squeeze(0).clone()
        labels = self.mask_label(prompt, labels, indices)
        # TODO: labels pad上IGNORE_TOKEN_ID
        # labels[:len(encoded_prompt) + 1] = IGNORE_TOKEN_ID # 这里不 + 1抵消bos，是因为可能最后一个token是空格，和回答的第一个token合在一起
        return {
            "input_ids": encoded_text["input_ids"].squeeze(0),
            "attention_mask": encoded_text["attention_mask"].squeeze(0),
            "labels": labels
        }
