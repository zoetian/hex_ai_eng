"""Compare two eval runs case by case: which prompts flipped pass<->fail.

Usage: uv run python scripts/diff_runs.py logs/run_A/evals_hard logs/run_B/evals_hard
"""

import json
import sys
from pathlib import Path


def load(d: str) -> dict[str, dict]:
    out = {}
    for p in Path(d).glob("*.json"):
        t = json.loads(p.read_text())
        out[t["case"]["prompt"]] = t
    return out


def main() -> None:
    a, b = load(sys.argv[1]), load(sys.argv[2])
    for label, cond in [
        ("FIXED (fail -> pass)", lambda x, y: not x and y),
        ("REGRESSED (pass -> fail)", lambda x, y: x and not y),
        ("STILL FAILING", lambda x, y: not x and not y),
    ]:
        rows = [k for k in a if k in b and cond(a[k]["result"]["passed"], b[k]["result"]["passed"])]
        print(f"\n## {label}: {len(rows)}")
        for k in rows:
            print(" -", k[:140])


if __name__ == "__main__":
    main()
