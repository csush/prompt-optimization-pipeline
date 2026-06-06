"""Load a small slice of GSM8K from the public OpenAI repo.

GSM8K lines look like:
    {"question": "...", "answer": "multi-line reasoning ...\\n#### 42"}
The gold final answer is the text after the last '####'.
"""

from __future__ import annotations

import json
import random
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .config import Config

_RAW_BASE = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data"
_CACHE = Path(__file__).resolve().parents[2] / "data"


@dataclass(frozen=True)
class Example:
    question: str
    gold: str  # final numeric answer, e.g. "42"
    solution: str  # full reference reasoning + answer


def _download(split: str) -> Path:
    """Cache the raw jsonl split locally; return its path."""
    _CACHE.mkdir(parents=True, exist_ok=True)
    dest = _CACHE / f"gsm8k_{split}.jsonl"
    if not dest.exists():
        urllib.request.urlretrieve(f"{_RAW_BASE}/{split}.jsonl", dest)
    return dest


def _parse_line(line: str) -> Example:
    row = json.loads(line)
    answer = row["answer"]
    gold = answer.split("####")[-1].strip().replace(",", "")
    return Example(question=row["question"], gold=gold, solution=answer)


def _load_split(split: str, n: int, seed: int) -> list[Example]:
    path = _download(split)
    with path.open() as f:
        rows = [_parse_line(ln) for ln in f if ln.strip()]
    random.Random(seed).shuffle(rows)
    return rows[:n]


def load_datasets(cfg: Config) -> tuple[list[Example], list[Example], list[Example]]:
    """Return (train, val, test) example lists.

    train/val are disjoint slices of the GSM8K train split; test comes from the
    held-out GSM8K test split so reported accuracy reflects generalization.
    """
    train_pool = _load_split("train", cfg.train_size + cfg.val_size, cfg.seed)
    train = train_pool[: cfg.train_size]
    val = train_pool[cfg.train_size : cfg.train_size + cfg.val_size]
    test = _load_split("test", cfg.test_size, cfg.seed)
    return train, val, test
