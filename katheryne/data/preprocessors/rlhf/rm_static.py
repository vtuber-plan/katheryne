# coding=utf-8
# Copyright 2024 XiaHan
# 
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from typing import Any, Dict

def fn_rm_static(sample: Dict[str, Any]) -> Dict[str, Any]:
    prompt: str = sample["prompt"]
    lines = prompt.splitlines()

    messages = []
    for line in lines:
        if len(line.strip()) == 0:
            continue
        line = line.strip()

        if line.startswith("Human:"):
            if len(messages) != 0 and messages[-1]["role"] == "user":
                messages[-1]["content"] += "\n" + line
            else:
                messages.append({
                    "role": "user",
                    "content": line.lstrip(),
                })
        elif line.startswith("Assistant:"):
            if len(messages) != 0 and messages[-1]["role"] == "assistant":
                messages[-1]["content"] += "\n" + line
            else:
                messages.append({
                    "role": "assistant",
                    "content": line.lstrip(),
                })
        else:
            if len(messages) != 0:
                messages[-1]["content"] += "\n" + line
            else:
                raise Exception("fn_rm_static mapping invalid data.")
            
    sample["messages"] = messages
    return sample
