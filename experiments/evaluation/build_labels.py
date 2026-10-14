"""Labelling helper: print the candidate frames a query actually surfaces.

This is **annotation tooling, not an evaluation**. It exists to make the manual
labelling step practical: a human needs to see which moments the retriever
believes are relevant in order to record the moment that is *actually* correct.

Why this cannot be automated
----------------------------
The obvious shortcut - take the top-ranked frame as the ground truth - would
grade the system on its own preferences. Any query the retriever already
handles well would be marked correct by construction, and every query it fails
would have no correct answer recorded. The resulting metrics would be
meaningless and, worse, would look legitimate.

What this tool does instead
---------------------------
* Runs the **real** retrieval pipeline against the **real** index.
* Prints a table of candidate frames: timestamp, thumbnail path, score.
* Marks nothing as correct. The human picks the true timestamp and writes it
  into the dataset CSV.

Usage
-----
    python -m experiments.evaluation.build_labels \\
        --video-id <VIDEO_UUID> \\
        --queries experiments/datasets/evaluation.csv \\
        --candidates 20
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path
from typing import List, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_BACKEND = _REPO_ROOT / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from experiments.evaluation.dataset import DatasetError, load_dataset  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m experiments.evaluation.build_labels",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--queries", required=True, help="Dataset CSV with the query column.")
    parser.add_argument(
        "--video-id",
        default=None,
        help="Restrict candidates to one video. Omit to search all indexed videos.",
    )
    parser.add_argument("--candidates", type=int, default=10, help="Frames to show per query.")
    parser.add_argument("--top-k", type=int, default=None, help="Override the candidate pool.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        dataset = load_dataset(args.queries)
    except DatasetError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    try:
        video_id = uuid.UUID(args.video_id) if args.video_id else None
    except ValueError:
        print(f"ERROR: --video-id '{args.video_id}' is not a valid UUID", file=sys.stderr)
        return 2

    from app.db.session import SessionLocal
    from app.schemas.graph import GraphSearchRequest
    from app.services.graph_search_service import graph_search

    db = SessionLocal()
    try:
        for query in dataset.queries:
            request = GraphSearchRequest(
                query=query.query,
                video_id=video_id or query.video_id,
                top_k=args.top_k or max(args.candidates, 10),
                # Depth 0 keeps the candidate list purely vector-driven, so the
                # labelling aid is not influenced by the graph ranking it is
                # meant to help evaluate.
                graph_depth=0,
                min_score=-1.0,
            )
            try:
                response = graph_search(db, request)
            except Exception as exc:  # noqa: BLE001
                print(f"\n=== {query.id}: {query.query}")
                print(f"  retrieval failed: {exc}")
                continue

            print(f"\n=== {query.id}: {query.query}")
            if not response.results:
                print("  no candidates returned - is a video indexed and status='ready'?")
                continue
            print(
                f"  {len(response.results)} candidate(s). "
                f"Watch each frame and record the second at which the query is "
                f"TRULY answered:"
            )
            print(f"  {'rank':>4}  {'timestamp':>10}  {'vector':>7}  thumbnail")
            for rank, result in enumerate(response.results[: args.candidates], start=1):
                print(
                    f"  {rank:>4}  {result.timestamp_seconds:>10.2f}  "
                    f"{result.vector_score:>7.4f}  {result.thumbnail_url}"
                )
    finally:
        db.close()

    print(
        "\nReminder: the correct timestamp is a judgement only you can make. "
        "Copy the timestamp you observed into the dataset's "
        "ground_truth_timestamp column. Do not simply copy rank 1 - that would "
        "grade the retriever on its own output."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
