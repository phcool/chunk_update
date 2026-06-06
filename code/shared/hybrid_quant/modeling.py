from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import shutil
from typing import Tuple

os.environ["HF_HOME"] = "/home/vrintern/tmp/.hf-cache"
os.environ["HF_XET_CACHE"] = "/home/vrintern/tmp/.hf-xet-cache"
os.environ["HF_MODULES_CACHE"] = "/home/vrintern/tmp/.hf-modules-cache"
os.environ["XDG_CACHE_HOME"] = "/home/vrintern/tmp/.cache"

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .cache import make_attention_cache


@dataclass
class RuntimeCaches:
    past_key_values: object
    fla_past_key_values: object
    mamba_inference_params: object


def load_nemotron(model_path: str, device: str, dtype: str = "bfloat16", attn_implementation: str = "flash_attention_2"):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(torch.device(device))
        torch.cuda.init()
    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }[dtype.lower()]
    _ensure_local_dynamic_modules(model_path)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.attn_implementation_new = attn_implementation
    hf_attn_implementation = attn_implementation
    if attn_implementation == "fused_mha":
        hf_attn_implementation = "sdpa"
    config.attn_implementation = hf_attn_implementation
    config._attn_implementation = hf_attn_implementation
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model.to(torch.device(device))
    model.eval()
    _patch_fla_device_context(force_cuda=str(device).startswith("cuda"))
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def _ensure_local_dynamic_modules(model_path: str):
    src = Path(model_path)
    if not src.exists():
        return
    cache_root = Path(os.environ.get("HF_MODULES_CACHE", "/home/vrintern/tmp/.hf-modules-cache"))
    dst = cache_root / "transformers_modules" / src.name
    dst.mkdir(parents=True, exist_ok=True)
    for parent in (cache_root, cache_root / "transformers_modules", dst):
        init_file = parent / "__init__.py"
        if not init_file.exists():
            init_file.write_text("", encoding="utf-8")
    for py_file in src.glob("*.py"):
        shutil.copy2(py_file, dst / py_file.name)


def _patch_fla_device_context(force_cuda: bool = False):
    try:
        import fla.utils as fla_utils
    except Exception:
        return
    if not force_cuda and not torch.cuda.is_available():
        return
    fla_utils.device = "cuda"
    fla_utils.device_platform = "cuda"
    fla_utils.device_name = "cuda"
    fla_utils.device_torch_lib = torch.cuda
    fla_utils.IS_NVIDIA = True
    fla_utils.IS_AMD = False
    fla_utils.IS_INTEL = False

    def _cuda_device_ctx(index: int):
        return torch.cuda.device(index if index is not None else torch.cuda.current_device())

    fla_utils.custom_device_ctx = _cuda_device_ctx


def new_runtime_caches(model, batch_size: int, max_seqlen: int, kv_mode: str, kv_group_size: int = 32) -> RuntimeCaches:
    _, fla_past_key_values, mamba_inference_params = model.get_init_cache(max_seqlen=max_seqlen, batch_size=batch_size)
    past_key_values = make_attention_cache(
        model.config,
        batch_size=batch_size,
        kv_mode=kv_mode,
        device=model.device,
        group_size=kv_group_size,
    )
    return RuntimeCaches(
        past_key_values=past_key_values,
        fla_past_key_values=fla_past_key_values,
        mamba_inference_params=mamba_inference_params,
    )


def prepare_attention_kv_caches(model, batch_size: int, max_seqlen: int):
    for layer in getattr(model.model, "layers", []):
        attn = getattr(layer, "self_attn", None)
        if attn is not None and hasattr(attn, "init_kv_cache"):
            attn.init_kv_cache(max_batch_size=batch_size, max_seq_len=max_seqlen)


def reset_model_sequence_state(model):
    if hasattr(model, "model") and hasattr(model.model, "has_previous_state"):
        model.model.has_previous_state = False
