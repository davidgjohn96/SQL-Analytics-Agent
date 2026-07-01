"""Prompt-comparison evaluation harness.

Runs the agent over evals/dataset.csv with two prompt variants and compares:

  * SQL validity        — did the generated SQL pass validation?
  * Execution success    — did it run without a DB error?
  * Answer quality       — does the result set match the expected query's result?
  * Latency              — wall-clock seconds per question.

Every run goes through the fully-instrumented agent, so each question also
produces a trace in Arize AX (grouped under the configured project).

Usage (from the sql-agent/ directory, with OPENAI_API_KEY set):

    python -m evals.experiments                 # run both variants A and B
    python -m evals.experiments --variants B    # run a single variant
    python -m evals.experiments --limit 5       # quick smoke run

Outputs (written to evals/):
    results_<variant>.csv          per-question detail
    runs_<variant>.json            Arize-experiment run format (push with `ax`)
    comparison.csv                 side-by-side aggregate metrics

To push a run to Arize as an experiment (see the arize-experiment skill):
    ax experiments create --name "promptA" --dataset sql-agent-evals \\
        --space "$ARIZE_SPACE_ID" --file evals/runs_A.json
"""

from __future__ import annotations

import argparse
import json
import os
import time

import pandas as pd
from dotenv import load_dotenv

load_dotenv(override=True)

from agent import tools  # noqa: E402
from agent.graph import run_agent  # noqa: E402
from agent.observability import flush  # noqa: E402

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(EVAL_DIR, "dataset.csv")


def _normalize_rows(columns: list[str], rows: list[list]) -> set[tuple]:
    """Order-insensitive, float-rounded representation of a result set."""
    norm = set()
    for row in rows:
        cells = []
        for cell in row:
            if isinstance(cell, float):
                cells.append(round(cell, 2))
            else:
                cells.append(cell)
        norm.add(tuple(cells))
    return norm


def _result_matches(state: dict, expected_sql: str) -> bool:
    """True if the agent's result set equals the expected query's result set."""
    if state.get("execution_error") or not state.get("columns"):
        return False
    expected = tools.execute_sql(expected_sql)
    if expected.error:
        return False  # expected SQL itself failed — can't judge
    got = _normalize_rows(state.get("columns", []), state.get("rows", []))
    want = _normalize_rows(expected.columns, expected.rows)
    return got == want


def run_variant(variant: str, df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, ex in df.iterrows():
        question = ex["question"]
        start = time.time()
        try:
            state = run_agent(question, prompt_variant=variant)
            error = None
        except Exception as exc:  # noqa: BLE001
            state, error = {}, str(exc)
        latency = round(time.time() - start, 3)

        valid = state.get("validated_sql") is not None and not error
        executed = valid and not state.get("execution_error")
        correct = bool(executed) and _result_matches(state, ex["expected_sql"])

        records.append(
            {
                "id": ex["id"],
                "question": question,
                "variant": variant,
                "generated_sql": state.get("generated_sql", ""),
                "valid": bool(valid),
                "executed": bool(executed),
                "correct": bool(correct),
                "latency_s": latency,
                "error": error
                or state.get("validation_error")
                or state.get("execution_error"),
            }
        )
        flag = "OK " if correct else ("RUN" if executed else ("VAL" if valid else "ERR"))
        print(f"  [{variant}] {ex['id']} {flag} {latency:>5.2f}s  {question[:50]}")

    return pd.DataFrame(records)


def to_arize_runs(detail: pd.DataFrame) -> list[dict]:
    """Convert per-question detail to the Arize experiment run schema."""
    runs = []
    for _, r in detail.iterrows():
        runs.append(
            {
                "example_id": r["id"],
                "output": r["generated_sql"],
                "evaluations": {
                    "sql_valid": {
                        "label": "valid" if r["valid"] else "invalid",
                        "score": 1.0 if r["valid"] else 0.0,
                    },
                    "execution_success": {
                        "label": "success" if r["executed"] else "failure",
                        "score": 1.0 if r["executed"] else 0.0,
                    },
                    "answer_correct": {
                        "label": "correct" if r["correct"] else "incorrect",
                        "score": 1.0 if r["correct"] else 0.0,
                    },
                },
                "metadata": {
                    "model": os.getenv("OPENAI_MODEL", "openai/gpt-oss-120b"),
                    "prompt_variant": r["variant"],
                    "latency_ms": int(r["latency_s"] * 1000),
                },
            }
        )
    return runs


def summarize(detail: pd.DataFrame, variant: str) -> dict:
    n = len(detail)
    return {
        "variant": variant,
        "n": n,
        "sql_validity": round(detail["valid"].mean(), 3),
        "execution_success": round(detail["executed"].mean(), 3),
        "answer_quality": round(detail["correct"].mean(), 3),
        "avg_latency_s": round(detail["latency_s"].mean(), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt-comparison experiments.")
    parser.add_argument(
        "--variants", default="A,B", help="Comma-separated variants to run (default A,B)."
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit number of questions.")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set. Set it before running experiments.")

    df = pd.read_csv(DATASET_PATH)
    if args.limit:
        df = df.head(args.limit)

    variants = [v.strip().upper() for v in args.variants.split(",") if v.strip()]
    summaries = []

    for variant in variants:
        print(f"\n=== Running variant {variant} on {len(df)} questions ===")
        detail = run_variant(variant, df)

        detail.to_csv(os.path.join(EVAL_DIR, f"results_{variant}.csv"), index=False)
        with open(os.path.join(EVAL_DIR, f"runs_{variant}.json"), "w") as fh:
            json.dump(to_arize_runs(detail), fh, indent=2)

        summaries.append(summarize(detail, variant))

    comparison = pd.DataFrame(summaries)
    comparison.to_csv(os.path.join(EVAL_DIR, "comparison.csv"), index=False)

    print("\n=== Comparison ===")
    print(comparison.to_string(index=False))

    flush()  # ship any buffered spans to Arize before exit


if __name__ == "__main__":
    main()
