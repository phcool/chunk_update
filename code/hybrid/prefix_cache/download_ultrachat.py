from __future__ import annotations

import argparse
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_HF_HOME = REPO_ROOT / ".hf_home"
DEFAULT_OUTPUT = REPO_ROOT / "datasets" / "ultrachat_200k"


def configure_hf_env(hf_home: Path) -> None:
    hf_home = hf_home.resolve()
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_DATASETS_CACHE"] = str(hf_home / "datasets")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_home / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(hf_home / "transformers")
    os.environ.pop("HF_HUB_CACHE", None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="HuggingFaceH4/ultrachat_200k")
    parser.add_argument("--split", default="train_sft")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "train_sft")
    parser.add_argument("--hf-home", type=Path, default=DEFAULT_HF_HOME)
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()

    configure_hf_env(args.hf_home)

    from datasets import load_dataset

    cache_dir = Path(os.environ["HF_DATASETS_CACHE"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.dataset, split=args.split, cache_dir=str(cache_dir))
    if args.max_rows > 0:
        ds = ds.select(range(min(args.max_rows, len(ds))))
    ds.save_to_disk(str(args.output))

    print(f"saved_to={args.output}")
    print(f"rows={len(ds)}")
    print(f"hf_home={os.environ['HF_HOME']}")
    print(f"datasets_cache={os.environ['HF_DATASETS_CACHE']}")


if __name__ == "__main__":
    main()
