from typing import List

from models.net import Net
from models.tiny_CNN import tinyCNN

def load_model(
    model_name:str,
    input_spec: List,
    )->Net:
    if model_name == "tinyCNN":
        return tinyCNN(input_spec=input_spec)
    else: 
        raise NotImplementedError(f"model {model_name} is not implemented")
