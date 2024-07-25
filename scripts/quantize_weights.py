import argparse
import os
import re
import sys
from fms.models import llama # registers adapters
from fms.utils.serialization import _get_adapter
import torch
from tqdm import tqdm
from fms.utils import serialization #load_state_dict
from safetensors.torch import save_file

from fms.modules.quantized import quant_dtype_to_torch_dtype
from fms.modules.rotated import full_normed_right_hadamard
from fms.utils.special_had import get_hadK

def quantize(weight: torch.Tensor, qdtype, bits, scaledtype, dtype=torch.float16, dim=-1, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    if device is None:
        device = weight.device
    
    
    max_qint = 2 ** (bits - 1) - 1
    min_qint = -(max_qint + 1)
    
    mag, _ = weight.abs().max(dim=dim, keepdim=True)
    mag = mag.clamp(min=1e-5)
    scale = (mag / max_qint).to(scaledtype)

    err_shape = list(weight.shape)
    err_shape[dim] = 1 # weight shape except for dim, since only 1 scale along dim
    best_err = torch.full(err_shape, torch.inf, device=device)
    steps = 100
    min_frac = 0.2
    for i in range(int((1-min_frac) * steps)): # 0.2 * 100: min % to shrink to, how many steps to cut max into
        mag1 = (1 - i / steps) * mag

        scale1 = mag1 / max_qint # has right shape based on dim
        weight1 = weight / scale1
        if qdtype in [torch.int8, torch.int16]:
            weight1 = weight1.clamp(min=min_qint, max=max_qint).round()
        # if sym:
        weight1 = weight1.type(qdtype)
        # else:
        #     weight1 = weight1.type(uqdtype)
        weight1_comp = (scale1 * (weight1)).to(dtype) # TODO: consider fp64 to compare

        diff = weight - weight1_comp
        err = diff.abs().pow(2.4).sum(dim, keepdim=True)

        # update best scale and err, per row
        is_new_best = err < best_err
        best_err = torch.where(is_new_best, err, best_err)
        scale = torch.where(is_new_best, scale1, scale)

    weight = weight / scale
    weight = weight.clamp(min=min_qint, max=max_qint).round()
    weight = weight.type(qdtype)
    return weight.to(device), scale.to(device)

def load_quantize_store(load_path: str, save_path: str, source: str, quant_dtype_str: str, rotate: bool):
    save_end = "weights.safetensors"
    if not save_path.endswith("/" + save_end): # TODO: clean up
        if save_path.endswith("/"):
            save_path = save_path + save_end
        else:
            save_path = save_path + "/" + save_end

    quant_dtype, bits = quant_dtype_to_torch_dtype(quant_dtype_str)

    # TODO: support distributed, etc.? function accepts those args
    print(f"Loading [lazy] state dict from: {load_path}")
    lazy_sd = serialization.load_state_dict(load_path)
    save_sd = {}

    # 1. Get the adapter from checkpoint sd to fms sd
    adapter = _get_adapter("llama", source)
    lazy_sd = adapter(lazy_sd)
    # # name map to avoid touching all lazy tensors
    # name_map = {key: key for key in lazy_sd.keys()}
    # name_map = adapter(name_map) # get new names
    # name_map = {name_old: name_new for name_new, name_old in name_map} # swap map

    def match(match_list: list[str], target: str):
        """Check if any of `match_list` is in `target`"""
        for m in match_list:
            if m in target:
                return m # TODO: ugly, but string should evaluate to True
        return False

    # norms = {
    #     "dec_norm": {},
    #     "ln": {},
    #     "ff_ln": {},
    # }
    quantize_match = [
        # "query"
        "qkv_fused",
        # "key",
        # "value",
        "dense",
        # "wg",
        # "w1",
        "wg1_fused",
        "w2",
    ]

    # values are things that are absorbed into keys
    absorptions = {
        "attn.in_proj.qkv_fused": "ln",
        "ff_sub_layer.wg1_fused": "ff_ln",
        "shared.head": "dec_norm"
    }
    absorb_keys = absorptions.keys()
    # dependency_vals = dependencies.values()
    # dependency_sd = {}

    print("Quantizing state dict...")
    for item, val in tqdm(lazy_sd.items()):
        # # if something will absorb this, set it aside for now
        # if item in dependency_vals and val is not None:
        #     dependency_sd[item] = val
        #     continue

        if rotate:
            # absorb if needed
            if match(absorb_keys, item):
                # dependency_num = re.findall(r'\.\d+\.', item)
                # if len(dependency_num) == 0:
                dependence = match(absorb_keys, item)
                dependency_name = item.replace(dependence, absorptions[dependence])
                # val2 = dependency_sd.pop(dependency_name, None)
                # if val2 is None:
                val2 = lazy_sd.get(dependency_name)
                # else:
                #     lazy_sd[dependency_name] = None # mark as done

                
                # absorb, works in all 3
                val = (val2.view(1, -1).to(torch.float64) * val.to(torch.float64)).to(val.dtype)


        # item = name_map[item_old]
        if match(quantize_match, item):
            assert item.endswith('.weight')
            item_s = item.replace('.weight', '.weight_scale')
            valq, val_s = quantize(val.to('cuda'), quant_dtype, bits, torch.float16, dim=-1)
            valq, val_s = valq.to('cpu'), val_s.to('cpu')
            save_sd[item] = valq
            save_sd[item_s] = val_s
        else:
            save_sd[item] = val

    print(f"Saving state dict to: {save_path}")
    os.makedirs(save_path[:save_path.rindex('/')], exist_ok=True)
    save_file(save_sd, save_path)
    print("Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to quantize weights and save"
    )
    parser.add_argument(
        "--load_path",
        type=str,
        required=True,
        help="Path to the directory containing LLaMa weights",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path to the directory containing LLaMa weights",
    )
    parser.add_argument(
        "--model_source",
        type=str,
        required=True,
        help="Source of the checkpoint. E.g. 'meta', 'hf', None",
    )
    parser.add_argument(
        "--quant_dtype",
        type=str,
        help="enables quantization to the specified dtype",
        default="",
        choices=["", "int8", "int4-fake"],
    )
    parser.add_argument(
        "--rotate",
        action="store_true",
    )

    args = parser.parse_args()

    load_quantize_store(args.load_path, args.save_path, args.model_source, args.quant_dtype, args.rotate)
