from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List

os.environ["HF_HOME"] = "/home/vrintern/tmp/.hf-cache"
os.environ["HF_DATASETS_CACHE"] = "/home/vrintern/tmp/.hf-cache/datasets"
os.environ["XDG_CACHE_HOME"] = "/home/vrintern/tmp/.cache"

from datasets import load_dataset


def load_wikitext_texts(dataset_path: str, split: str = "test") -> List[str]:
    path = Path(dataset_path)
    if path.exists():
        ds = load_dataset(str(path), "wikitext-103-raw-v1", split=split)
    else:
        ds = load_dataset(dataset_path, "wikitext-103-raw-v1", split=split)
    return [x for x in ds["text"] if x and not x.isspace()]


def token_segments(tokenizer, texts: Iterable[str], context_length: int, max_eval_tokens: int):
    joined = "\n\n".join(texts)
    ids = tokenizer(joined, add_special_tokens=False, return_tensors="pt").input_ids[0]
    usable = min(ids.numel() - 1, max_eval_tokens)
    pos = 0
    while pos < usable:
        end = min(pos + context_length + 1, usable + 1)
        if end - pos >= 2:
            yield ids[pos:end]
        pos = end - 1
