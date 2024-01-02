from typing import Union
from pydantic import BaseModel, Field

class DatasetPath(BaseModel):
    path: str
    sample: Union[int, float] = 1.0
    shuffle: bool = False

    def __str__(self) -> str:
        return self.path