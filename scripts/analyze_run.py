"""Summarize an eval run's traces: pass/fail per case, failure types, tool usage.

Usage: uv run python scripts/analyze_run.py [logs/run_XXXX/evals_hard] [--show-failures]
"""

import glob
import json
import sys
from collections import Counter
from pathlib import Path


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    run_dir = Path(args[0]) if args else Path(sorted(glob.glob("logs/run_*/evals_hard"))[-1])
    show = "--show-failures" in sys.argv
    traces = [json.loads(p.read_text()) for p in sorted(run_dir.glob("*.json"))]
    tool_counts: Counter[str] = Counter()
    failures = []
    for t in traces:
        tools = [e["data"]["name"] for e in t["events"] if e["type"] == "TOOL_CALL_PARSED"]
        tool_counts.update(tools)
        r = t["result"]
        if not r["passed"]:
            failures.append((t, r, tools))
    n = len(traces)
    print(f"{run_dir}: {n - len(failures)}/{n} passed")
    print("failure types:", Counter(r.get("failure_type") for _, r, _ in failures))
    print("tool calls:", dict(tool_counts))
    if show:
        for t, r, tools in failures:
            print("\n" + "=" * 100)
            print("PROMPT:", t["case"]["prompt"])
            print("GUIDES READ:", [
                e["data"]["arguments"].get("name") for e in t["events"]
                if e["type"] == "TOOL_CALL_PARSED" and e["data"]["name"] == "read_guide"
            ])
            print("ERROR:", r.get("failure_type"), (r.get("error") or "")[:300])
            print("GOLD:", t["case"]["gold_query"])
            print("SUBMITTED:", r.get("submitted_query"))


if __name__ == "__main__":
    main()
