# coding=utf-8
# Copyright 2024 XiaHan
# 
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from typing import Any, Dict

def fn_merge_fields(sample: Dict[str, Any]) -> Dict[str, Any]:
    out = []
    for subfield in sample.items():
        data = sample[subfield]
        if isinstance(data, str):
            line = data
        elif isinstance(data, list):
            line = "\n".join(data)
        else:
            line = str(data)
        out.append(line)
    return {"text": "\n".join(out)}