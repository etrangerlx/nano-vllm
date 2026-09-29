import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    prefix = getattr(model, "weight_prefix", None)
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                name = weight_name[len(prefix):] if prefix and weight_name.startswith(prefix) else weight_name
                for k in packed_modules_mapping:
                    if k in name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    try:
                        param = model.get_parameter(name)
                    except (KeyError, AttributeError):
                        continue  # skip weights without a matching module (e.g. visual / mtp)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
