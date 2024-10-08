from contextlib import nullcontext
from functools import partial
from typing import Any, Callable, MutableMapping, Optional

import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.distributed_c10d import ProcessGroup
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy

from fms import distributed
from fms.distributed.strategy import (
    TensorParallelStrategy,
    UniformModelParallelStrategy,
)
from fms.utils import serialization

__models: MutableMapping[str, MutableMapping[str, Callable[[], nn.Module]]] = {}


def register_model(architecture: str, variant: str, factory: Callable[[], nn.Module]):
    """
    Registers a model variant to be made available in the registration API.
    Args:
    architecture: The name of the model architecture, e.g. 'llama'
    variant: A reference for a particular configuration of the architecture,
        e.g. '7b'
    factory: A callable that constructs an instance of the model variant.
    """
    variants: MutableMapping[str, Callable[[], nn.Module]] = {}
    if architecture in __models:
        variants = __models[architecture]
    if variant in variants:
        raise KeyError(
            f"Variant {variant} already registered for architecture {architecture}"
        )
    variants[variant] = factory
    __models[architecture] = variants


def list_models():
    """
    Lists registered model architectures.
    """
    return list(__models.keys())


def list_variants(architecture: str):
    """
    Lists available variants (configurations) of a model architecture.
    E.g. `models.list_variants('llama')` -> ['micro', '7b', '13b', '70b']
    Args:
    architecture: one of the registered architectures returned by `list_models()`.
    """
    if architecture not in __models:
        raise KeyError(
            f"{architecture} is not registered. See `models.list_models()` for available architectures"
        )
    return list(__models[architecture].keys())


def _get_model_instance(
    architecture: str, variant: str, *, dtype=None, device=None, extra_args: dict = {}
) -> nn.Module:
    """
    Gets a model by name and variant, e.g. `models.get_model('llama', '7b')`
    Does not load weights.
    See public API `models.get_model()`
    Args:
    architecture: one of the architectures from list_models(). E.g. llama.
    variant: one of the variants from list_variants(architecture). E.g. '7b'
    extra_args: kwargs to be passed to the model factory.
    """
    if architecture not in __models:
        raise KeyError(
            f"{architecture} is not registered. See `models.list_models()` for available architectures"
        )
    if variant not in __models[architecture]:
        raise KeyError(
            f'{variant} is not a registered variant of {architecture}. See `models.list_variants("{architecture}")` for available variants.'
        )

    model_factory = __models[architecture][variant]

    orig = torch.get_default_dtype()

    try:
        if dtype is not None:
            torch.set_default_dtype(dtype)
        with device if device is not None else nullcontext():
            return model_factory(**extra_args)
    finally:
        torch.set_default_dtype(orig)


def _guess_num_layers(state_dict):
    """
    This function attempts to guess the number of "layers" in a state_dict by
    looking for lists of sub modules. This can be used to setup model-parallel
    when we don't yet have a model instance.
    """
    if state_dict is None or len(state_dict) == 0:
        raise ValueError(
            "Use model parallel with pre-trained models that have a state dict"
        )

    layers = set()
    import re

    for key in state_dict.keys():
        # when there's a list of layers, layers have numeric IDs in the key
        layerid = re.sub("[^.]*\\.([0-9]+)\\..*", "\\1", key)
        if layerid != key:
            layers.add(layerid)
    return len(layers)


def _class_hierarchy(clz):
    if clz == object:
        return {clz}
    bases = clz.__bases__
    all = [_class_hierarchy(c) for c in bases]
    result = {clz}
    for classes in all:
        result = result | classes
    return result


def _fsdp_autowrap_policy(module: nn.Module, recurse: bool, nonwrapped_numel: int):
    if recurse:
        return True
    classes = _class_hierarchy(module.__class__)
    for clz in classes:
        name = str(clz).lower()
        if ("layer" in name or "block" in name) and "layernorm" not in name:
            return True
    return False


def _activation_checkpoint_check_fn(layer):
    for name in layer.__class__.__bases__:
        name = str(name).lower()
        if "block" in name or "layer" in name:
            return True
    return False


def _fsdp_wrap(
    model: nn.Module,
    distributed_strategy: Optional[str],
    device: torch.device,
    rank0: bool,
) -> nn.Module:
    # initializes parameters that are on meta devices
    def init_fn(x: nn.Module):
        if not rank0:
            return x.to_empty(device=device, recurse=False)
        else:
            return x

    # TODO: enable other policies
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    if distributed_strategy == "fsdp":
        dp_strategy = ShardingStrategy.FULL_SHARD
    elif distributed_strategy == "hsdp":
        dp_strategy = ShardingStrategy.HYBRID_SHARD
    elif distributed_strategy == "ddp":
        dp_strategy = ShardingStrategy.NO_SHARD
    else:
        raise KeyError("distributed strategy should be one of fsdp, dpp, or hsdp")

    model = FSDP(
        model,
        param_init_fn=init_fn,
        sync_module_states=True,
        device_id=device.index,
        limit_all_gathers=True,
        auto_wrap_policy=_fsdp_autowrap_policy,
        mixed_precision=mp_policy,
        sharding_strategy=dp_strategy,
    )

    wrapper_fn = partial(
        checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
    )
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=wrapper_fn,
        check_fn=_activation_checkpoint_check_fn,
    )

    return model

def _quantize_inplace(model: nn.Module, qdtype_str: str, rotate: bool, device_type: str, activ_clip_ratio: Optional[float], kv_clip_ratio: Optional[float]) -> None:
    from fms.models.llama import LLaMA, LLaMABlock
    from fms.modules import quantized
    from fms.modules import rotated
    from fms.modules.quantized import quant_dtype_to_torch_dtype
    if not isinstance(model, LLaMA):
        raise ValueError("quantized model only supported for LLaMa")

    # TODO: prevent the model from initializing the full unquantized size in VRAM to allow it to initialize on small GPUs which only fit the quantized version

    quant_dtype, bits = quant_dtype_to_torch_dtype(qdtype_str)

    if activ_clip_ratio is None:
        activ_clip_ratio = 1
    if kv_clip_ratio is None:
        kv_clip_ratio = 1

    def swap_linear(old: nn.Linear, had_size: Optional[int] = None, completed_size: Optional[int] = None):
        if had_size:
            is_full_had = not completed_size
            return rotated.Linear(old.in_features, old.out_features, quant_dtype, bits, activ_clip_ratio, is_full_had, had_size, completed_size, old.bias, old.weight.device)
        return quantized.Linear(old.in_features, old.out_features, quant_dtype, bits, activ_clip_ratio, old.bias, old.weight.device)

    def quant_reset_parameters(module):
        for m in module.modules():
            if isinstance(m, nn.Linear):
                m.weight.zero_()

    if rotate:
        model.dec_norm.elementwise_scale = False
        old_pos_enc = model.rot_emb
        emb_kq_per_head = model.config.emb_dim // model.config.nheads
        model.rot_emb = rotated.RotaryEmbedding(emb_kq_per_head, old_pos_enc.dim, old_pos_enc.ratio, old_pos_enc.max_seq_len, old_pos_enc.ntk_scaling)
        model.rot_emb.cached_freqs = old_pos_enc.cached_freqs
        model.rot_emb.max_seq_len_cached = old_pos_enc.max_seq_len_cached

    kv_quantizer = quantized.KVCacheQuantizer(quantized.signed_to_unsigned_dtype(quant_dtype), bits, kv_clip_ratio)

    for layer in model.layers:
        layer: LLaMABlock
        if rotate:
            layer.ln.elementwise_scale = False
            layer.ff_ln.elementwise_scale = False
        
        attn = layer.attn
        attn.kv_quantizer = kv_quantizer
        attn.dense = swap_linear(attn.dense, *((attn.nheads, attn.emb_v_per_head) if rotate else (None, None))) # online partial rotation before dense (per head rotation is already done by weight)
        if rotate:
            old_pos_enc = attn.position_encoder
            attn.position_encoder = model.rot_emb
        if attn.fused:
            attn.in_proj.qkv_fused = swap_linear(attn.in_proj.qkv_fused)
        else:
            attn.in_proj.query = swap_linear(attn.in_proj.query)
            attn.in_proj.key = swap_linear(attn.in_proj.key)
            attn.in_proj.value = swap_linear(attn.in_proj.value)
        attn.reset_parameters = partial(quant_reset_parameters, module=attn)
        
        ff = layer.ff_sub_layer
        ff.reset_parameters = partial(quant_reset_parameters, module=ff)
        ff.wg1_fused = swap_linear(ff.wg1_fused)
        ff.w2 = swap_linear(ff.w2, ff.w2.in_features if rotate else None) # online full rotation before down proj
    if device_type == "cuda":
        # free up mem saved by quantization
        torch.cuda.empty_cache()

def _is_dp(distributed_strategy):
    return distributed_strategy in {"fsdp", "hsdp", "ddp"}


def get_model(
    architecture: str,
    variant: str,
    model_path: Optional[str] = None,
    source: Optional[str] = None,
    device_type: str = "cpu",
    distributed_strategy: Optional[str] = None,
    checkpoint_sharding: Optional[str] = None,
    group: Optional[ProcessGroup] = None,
    quant_dtype: Optional[str] = None,
    activ_clip_ratio: Optional[float] = None,
    kv_clip_ratio: Optional[float] = None,
    rotate: bool = False,
    **kwargs,
):
    """
    Load an instance of a model with weights.

    Args:
    architecture: the model architecture, e.g. llama. See
                `models.list_models()`.
    variant: the configuration of the model, e.g. 7b. See
                `models.list_variants(architecture)`
    model_path: the path to the state_dict of weights. If None, don't load.
    device_type: where to load the model
    distributed_strategy: None, 'fsdp', 'hsdp', 'tp', or 'mp'.
    checkpoint_sharding: how the checkpoint files are sharded: None, 'tp',
                'fsdp', or 'layer'. If None, guess based on files.
    source: If the weights in the state dict didn't come from an FMS model,
                `source` specifies which conversion function might be needed.
                See `serialization.list_sources(architecture)`
    group: ProcessGroup The PG to use for any model distribution
    """
    rank, world_size = distributed.rank_and_world(group)
    local_rank = distributed.local_rank()

    if distributed_strategy is None or distributed_strategy == "":
        if world_size > 1:
            distributed_strategy = "tp"

    if device_type == "cuda":
        device = torch.device(device_type, local_rank)
    else:
        device = torch.device(device_type)

    hsdp = distributed_strategy == "hsdp"
    fsdp = distributed_strategy == "fsdp"
    ddp = distributed_strategy == "ddp"
    if hsdp or fsdp or ddp:
        if (hsdp and local_rank != 0) or ((fsdp or ddp) and rank != 0):
            initial_device = torch.device("meta")
        else:
            initial_device = torch.device("cpu")
    elif distributed_strategy == "mp":
        initial_device = torch.device("cpu")
    else:
        initial_device = device

    lazy_sd: MutableMapping[str, Any] = {}
    if model_path is not None:
        lazy_sd = serialization.load_state_dict(
            model_path,
            source=source,
            distributed_strategy=distributed_strategy,
            checkpoint_sharding=checkpoint_sharding,
            initial_device=initial_device,
            rank=rank,
            world_size=world_size,
        )

    extra_args = kwargs
    if "distributed_strategy" not in extra_args:
        if distributed_strategy == "tp":
            print("using tensor parallel")
            extra_args["distributed_strategy"] = TensorParallelStrategy()
        elif distributed_strategy == "mp":
            print("using model parallel")
            devices = [i for i in range(torch.cuda.device_count())]
            extra_args["distributed_strategy"] = UniformModelParallelStrategy(
                devices, _guess_num_layers(lazy_sd)
            )

    # Create the model
    fms_model = _get_model_instance(
        architecture, variant, device=initial_device, extra_args=extra_args
    )

    # Choose when to wrap and load the model weights based on the combination
    # distribution strategy and checkpoint sharding
    pre_load = (
        distributed_strategy in ["fsdp", "hsdp"] and checkpoint_sharding != "fsdp"
    )

    def model_wrap(model):
        if _is_dp(distributed_strategy):
            return _fsdp_wrap(model, distributed_strategy, device, rank == 0)
        return model

    if not pre_load:
        fms_model = model_wrap(fms_model)

    if quant_dtype:
        _quantize_inplace(fms_model, quant_dtype, rotate, device_type, activ_clip_ratio, kv_clip_ratio)

    if len(lazy_sd):
        serialization.load_state_dict_into_model(
            fms_model,
            lazy_sd,
            architecture,
            source if source is not None else "fms",
            distributed_strategy,
            checkpoint_sharding,
            initial_device,
        )
    elif hasattr(fms_model, "reset_parameters"):
        fms_model.reset_parameters()

    if pre_load:
        fms_model = model_wrap(fms_model)

    return fms_model


from fms.models import gpt_bigcode, llama, mixtral, roberta
