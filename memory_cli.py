from __future__ import annotations

import argparse
import json
from pathlib import Path

from memory import NLAMemoryStore


def _seed(mem: NLAMemoryStore, path: Path):
    items = json.loads(path.read_text(encoding="utf-8"))
    for item in items:
        mem.add_memory(item, require_confirmation=False)
    print(f"seed done: {len(items)} entries")


def _search(mem: NLAMemoryStore, query: str, top_k: int):
    result = mem.search_memory(query=query, top_k=top_k)
    print(mem.summarize_memory(result))


def _print_json(obj: dict):
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="NLA memory CLI")
    parser.add_argument("--seed", type=str, default="")
    parser.add_argument("--query", type=str, default="")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--list-pending", action="store_true")
    parser.add_argument("--approve", type=str, default="")
    parser.add_argument("--get", type=str, default="")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--cluster", action="store_true")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    mem = NLAMemoryStore()
    if args.seed:
        _seed(mem, Path(args.seed))
    if args.query:
        _search(mem, args.query, args.top_k)
    if args.list_pending:
        _print_json(mem.list_pending_memories(limit=args.limit))
    if args.approve:
        _print_json(mem.approve_pending_memory(args.approve))
    if args.get:
        _print_json(mem.get_memory(args.get, include_pending=True))
    if args.evaluate:
        _print_json(mem.evaluate_memory_quality())
    if args.cluster:
        _print_json({"status": "ok", "clusters": mem.cluster_similar_memories()})


if __name__ == "__main__":
    main()
