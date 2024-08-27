# coding=utf-8
# Copyright 2024 XiaHan
# 
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import json
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple
from pydantic import BaseModel

class DatasetTags(BaseModel):
    role_tag: Optional[str] = "from"
    content_tag: Optional[str] = "value"
    user_tag: Optional[str] = "human"
    assistant_tag: Optional[str] = "gpt"
    observation_tag: Optional[str] = "observation"
    function_tag: Optional[str] = "function_call"
    system_tag: Optional[str] = "system"

class DatasetColumn(BaseModel):
    prompt: Optional[str] = "instruction"
    query: Optional[str] = "input"
    response: Optional[str] = "output"
    history: Optional[str] = None
    messages: Optional[str] = "conversations"
    system: Optional[str] = None
    tools: Optional[str] = None
    images: Optional[str] = None
    chosen: Optional[str] = None
    rejected: Optional[str] = None
    kto_tag: Optional[str] = None


class DatasetInfo(BaseModel):
    hf_hub_url: Optional[str] = None
    ms_hub_url: Optional[str] = None
    script_url: Optional[str] = None
    file_name: Optional[str] = None
        
    formatting: Optional[str] = None
    ranking: Optional[bool] = None
    subset: Optional[str] = None
    split: Optional[str] = None
    folder: Optional[str] = None
    num_samples: Optional[int] = None
    columns: Optional[DatasetColumn] = None
    tags: Optional[DatasetTags] = None

class DatasetPool(object):
    def __init__(self):
        self.datasets: Dict[str, DatasetInfo] = {}
    
    @staticmethod
    def from_json_file(path: str) -> "DatasetPool":
        with open(path, "r", encoding="utf-8") as f:
            datasets_list = json.loads(f.read())
        lst = DatasetPool()
        for dataset_name, each_dict in datasets_list.items():
            lst[dataset_name] = DatasetInfo.model_validate(each_dict)
        return lst

    def items(self) ->  Iterable[Tuple[str, DatasetInfo]]:
        return self.datasets.items()