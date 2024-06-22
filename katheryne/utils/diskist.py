# coding=utf-8
# Copyright 2024 XiaHan
# 
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.


import os
import pickle
from typing import Any, List, Literal
import mmap
import json
import blosc
from tqdm import tqdm

META_FILE_NAME = "meta.json"
DATA_FILE_NAME = "data.pkl"

class DiskistMeta(object):
    def __init__(self) -> None:
        setattr(self, "version", "0.0.1")
        setattr(self, "data", [])
        setattr(self, "compression", False)
    
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)
    
    @staticmethod
    def read_from_json_file(path: str) -> "DiskistMeta":
        meta = DiskistMeta()
        meta.__dict__.clear()
        with open(path, "r", encoding="utf-8") as f:
            meta_data = json.loads(f.read())
            for key, value in meta_data.items():
                setattr(meta, key, value)
        return meta
    
    def write_to_json_file(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.__dict__))

def write_diskist(path: str, objs: List[Any], compression: bool=True) -> None:
    if os.path.exists(path):
        raise Exception("The diskist path contains files.")
    else:
        os.makedirs(path)
    meta = DiskistMeta()
    if compression:
        meta.compression = True
    with open(os.path.join(path, DATA_FILE_NAME), "wb") as f:
        for obj in objs:
            bytes_obj = pickle.dumps(obj, protocol=None)
            if compression:
                c_bytesobj = blosc.compress(bytes_obj, typesize=8)
                bytes_obj = c_bytesobj
            start_pos = f.tell()
            obj_len = len(bytes_obj)
            f.write(bytes_obj)
            meta.data.append((start_pos, obj_len))
    
    meta.write_to_json_file(os.path.join(path, META_FILE_NAME))

def extend_diskist(path: str, objs: List[Any]) -> None:
    if not os.path.exists(path):
        return write_diskist(path, objs)
    meta = DiskistMeta.read_from_json_file(os.path.join(path, META_FILE_NAME))
    with open(os.path.join(path, DATA_FILE_NAME), "r+b") as f:
        f.seek(0, os.SEEK_END)
        for obj in objs:
            bytes_obj = pickle.dumps(obj, protocol=None)
            if meta.compression:
                c_bytesobj = blosc.compress(bytes_obj, typesize=8)
                bytes_obj = c_bytesobj
            start_pos = f.tell()
            obj_len = len(bytes_obj)
            f.write(bytes_obj)
            meta.data.append((start_pos, obj_len))
    
    meta.write_to_json_file(os.path.join(path, META_FILE_NAME))
    
class Diskist(object):
    def __init__(self, path) -> None:
        self.path = path
        self.meta = DiskistMeta.read_from_json_file(os.path.join(path, META_FILE_NAME))

        self.data_f = open(os.path.join(path, DATA_FILE_NAME), "r+b")
        self.data_f.seek(0, os.SEEK_END)
        file_size = self.data_f.tell()
        self.data_f.seek(0, os.SEEK_SET)
        if file_size != 0:
            # memory-map the file, size 0 means whole file
            self.data_mm = mmap.mmap(self.data_f.fileno(), 0)
        else:
            self.data_mm = None

    def __len__(self):
        return len(self.meta.data)

    def __getitem__(self, idx):
        meta_obj = self.meta.data[idx]
        start_pos = meta_obj[0]
        obj_size = meta_obj[1]

        self.data_mm.seek(start_pos, os.SEEK_SET)
        c_bytes_obj = self.data_mm.read(obj_size)
        bytes_obj = blosc.decompress(c_bytes_obj)
        obj = pickle.loads(bytes_obj)
        return obj

    def close(self):
        if self.data_mm is not None:
            self.data_mm.close()
        self.data_f.close()
