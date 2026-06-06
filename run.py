"""Entry point: optimize a GSM8K system prompt for a local student model.

Usage:
    uv run run.py            # full POC run (sizes from env / defaults)
    uv run run.py --smoke    # tiny, fast sanity check
    uv run run.py --save out.txt   # also write the optimized prompt to a file
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from pipeline.config import Config  # noqa: E402
from pipeline.optimize import run  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GEPA prompt-optimization POC")
    p.add_argument("--smoke", action="store_true", help="tiny fast run")
    p.add_argument("--save", metavar="PATH", help="write optimized prompt to file")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = Config()
    if args.smoke:
        cfg = dataclasses.replace(
            cfg, train_size=3, val_size=2, test_size=3, max_metric_calls=8
        )

    report = run(cfg)

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"Baseline accuracy : {report.baseline_accuracy:.1%}")
    print(f"Optimized accuracy: {report.optimized_accuracy:.1%}  (n={report.n_test})")
    delta = report.optimized_accuracy - report.baseline_accuracy
    print(f"Delta             : {delta:+.1%}")
    print("\n--- Seed prompt ---\n" + report.seed_prompt)
    print("\n--- Optimized prompt ---\n" + report.best_prompt)

    if args.save:
        Path(args.save).write_text(report.best_prompt)
        print(f"\nSaved optimized prompt -> {args.save}")


if __name__ == "__main__":
    main()
