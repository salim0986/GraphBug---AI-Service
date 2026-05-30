"""
M9 — Graph Bug RAG evaluation harness.

Measures retrieval quality (Ragas, optional) and review quality (F1 over
expected findings) across four ablation modes:

  full       — full GraphRAG (graph + vector)
  no-graph   — vector only (no Neo4j context)
  no-vector  — graph only  (no Qdrant context)
  no-both    — no RAG at all (baseline)

Usage:
  # Offline (no LLM, scores F1=0 baseline — for CI)
  python evals/run_eval.py --offline

  # Live (requires GEMINI_API_KEY or ANTHROPIC_API_KEY etc.)
  python evals/run_eval.py

  # Single ablation mode
  python evals/run_eval.py --mode no-graph

  # Custom golden set + output path
  python evals/run_eval.py --golden evals/golden_prs.yaml --output evals/results/run1.csv

Exit code 1 if mean F1 (full mode) drops ≥5 percentage points below a
previous baseline stored at evals/results/baseline.csv (CI gate).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Make src importable when run from the project root or from evals/
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent))

from src.review_schema import Finding, ReviewOutput


# ============================================================================
# Data models
# ============================================================================

@dataclass
class ExpectedFinding:
    file: str
    category: str
    severity: str
    line: Optional[int] = None
    must_mention: List[str] = field(default_factory=list)


@dataclass
class GoldenPR:
    id: str
    title: str
    description: str
    files: List[Dict[str, Any]]
    expected_findings: List[ExpectedFinding]
    graphrag_context: Optional[Dict[str, Any]] = None


@dataclass
class EvalResult:
    pr_id: str
    mode: str
    # Ragas retrieval metrics (-1 = not available)
    context_precision: float = -1.0
    context_recall: float = -1.0
    answer_relevance: float = -1.0
    faithfulness: float = -1.0
    # Review quality
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    # Metadata
    findings_generated: int = 0
    findings_expected: int = 0


# ============================================================================
# Golden PR loading
# ============================================================================

def load_golden_prs(path: str) -> List[GoldenPR]:
    with open(path) as f:
        raw = yaml.safe_load(f)
    prs = []
    for entry in raw["prs"]:
        expected = [
            ExpectedFinding(
                file=ef["file"],
                category=ef["category"],
                severity=ef["severity"],
                line=ef.get("line"),
                must_mention=ef.get("must_mention", []),
            )
            for ef in entry.get("expected_findings", [])
        ]
        prs.append(GoldenPR(
            id=entry["id"],
            title=entry["title"],
            description=entry.get("description", ""),
            files=entry.get("files", []),
            expected_findings=expected,
            graphrag_context=entry.get("graphrag_context"),
        ))
    return prs


# ============================================================================
# F1 scoring
# ============================================================================

def finding_matches(generated: Finding, expected: ExpectedFinding) -> bool:
    """
    True if a generated finding corresponds to an expected one.

    Match rules (all must pass):
    1. File — generated.file must contain expected.file as a suffix
               (handles "src/auth.py" vs "auth.py")
    2. Category — exact match
    3. Line — within ±5 lines (skipped when either side has no line)
    4. must_mention — every keyword must appear in the description (case-insensitive)
    """
    # Rule 1: file
    gen_file  = generated.file.replace("\\", "/")
    exp_file  = expected.file.replace("\\", "/")
    if not (gen_file.endswith(exp_file) or exp_file.endswith(gen_file)
            or exp_file in gen_file or gen_file in exp_file):
        return False

    # Rule 2: category
    if generated.category.lower() != expected.category.lower():
        return False

    # Rule 3: line proximity
    if expected.line is not None and generated.line is not None:
        if abs(expected.line - generated.line) > 5:
            return False

    # Rule 4: must-mention keywords
    desc_lower = (generated.description or "").lower()
    if not all(kw.lower() in desc_lower for kw in expected.must_mention):
        return False

    return True


def score_review(review: ReviewOutput, expected: List[ExpectedFinding]) -> Dict[str, Any]:
    """Compute TP / FP / FN and derive precision, recall, F1."""
    tp = 0
    matched_expected: set = set()

    for gen_f in review.findings:
        for idx, exp_f in enumerate(expected):
            if idx not in matched_expected and finding_matches(gen_f, exp_f):
                tp += 1
                matched_expected.add(idx)
                break

    fp = len(review.findings) - tp
    fn = len(expected) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {
        "true_positives":  tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
    }


# ============================================================================
# Ragas (optional)
# ============================================================================

def ragas_metrics(
    question: str,
    answer: str,
    contexts: List[str],
    ground_truth: str,
) -> Dict[str, float]:
    """
    Run Ragas metrics if the library is available.
    Returns -1.0 for all metrics when Ragas is not installed or fails.
    """
    try:
        from ragas import evaluate                        # type: ignore[import]
        from ragas.metrics import (                       # type: ignore[import]
            context_precision,
            context_recall,
            answer_relevancy,
            faithfulness,
        )
        from datasets import Dataset                      # type: ignore[import]

        dataset = Dataset.from_dict({
            "question":     [question],
            "answer":       [answer],
            "contexts":     [contexts or ["(no context)"]],
            "ground_truth": [ground_truth],
        })
        result = evaluate(
            dataset,
            metrics=[context_precision, context_recall, answer_relevancy, faithfulness],
        )
        return {
            "context_precision": round(float(result["context_precision"]), 4),
            "context_recall":    round(float(result["context_recall"]),    4),
            "answer_relevance":  round(float(result["answer_relevancy"]),  4),
            "faithfulness":      round(float(result["faithfulness"]),      4),
        }
    except ImportError:
        return _ragas_unavailable()
    except Exception as exc:
        print(f"  [ragas] error: {exc}", flush=True)
        return _ragas_unavailable()


def _ragas_unavailable() -> Dict[str, float]:
    return {
        "context_precision": -1.0,
        "context_recall":    -1.0,
        "answer_relevance":  -1.0,
        "faithfulness":      -1.0,
    }


# ============================================================================
# Context building (ablation modes)
# ============================================================================

def build_contexts(pr: GoldenPR, no_graph: bool, no_vector: bool) -> List[str]:
    """
    Build the list of context strings for a given ablation mode.

    no_graph  — skip entity code snippets from Neo4j
    no_vector — skip similar-code results from Qdrant
    """
    if not pr.graphrag_context:
        return []

    ctx = pr.graphrag_context
    parts: List[str] = []

    if not no_graph:
        for entity in ctx.get("entities", []):
            name = entity.get("name", "")
            file = entity.get("file", "")
            code = entity.get("code", "")
            if name or code:
                parts.append(f"Entity `{name}` in {file}:\n{code}")

    if not no_vector:
        for sim in ctx.get("similar_code", []):
            file    = sim.get("file", "")
            line    = sim.get("line", "")
            snippet = sim.get("snippet", "")
            parts.append(f"Similar code at {file}:{line}:\n{snippet}")

    return parts


# ============================================================================
# Review generation
# ============================================================================

def _build_prompt(pr: GoldenPR, contexts: List[str]) -> str:
    files_text = "\n\n".join(
        f"### {f['filename']}\n```diff\n{f.get('patch', '').strip()}\n```"
        for f in pr.files
    )
    ctx_text = "\n\n".join(contexts) if contexts else "(None)"
    return (
        "## Code Review Request — JSON Output\n\n"
        f"**PR:** {pr.title}\n"
        f"**Description:** {pr.description or 'No description.'}\n\n"
        "### Files Changed\n"
        f"{files_text}\n\n"
        "### GraphRAG Context\n"
        f"{ctx_text}\n\n"
        "Return a JSON ReviewOutput with findings "
        "(severity, category, file, line, description, suggestion)."
    )


def generate_review_offline(_pr: GoldenPR, _contexts: List[str]) -> ReviewOutput:
    """Stub: returns an empty review for offline/CI mode (F1 = 0)."""
    return ReviewOutput(
        summary="[offline stub — no LLM called]",
        overall_assessment="comment",
        risk_level="low",
    )


def generate_review_live(pr: GoldenPR, contexts: List[str]) -> ReviewOutput:
    """Call a real LLM via LLMClient (requires API key env var)."""
    api_key  = (
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    provider = os.environ.get("EVAL_LLM_PROVIDER", "gemini")

    if not api_key:
        print("  [warn] No API key found — falling back to offline stub", flush=True)
        return generate_review_offline(pr, contexts)

    import asyncio
    from src.llm_client import LLMClient, LLMConfig

    client = LLMClient(LLMConfig(
        provider=provider,
        api_key=api_key,
        retry_attempts=2,
        retry_delay=1.0,
    ))
    prompt = _build_prompt(pr, contexts)
    return asyncio.run(client.generate_structured("flash", prompt, ReviewOutput))


# ============================================================================
# Evaluation loop
# ============================================================================

# Each tuple: (mode_name, no_graph, no_vector)
ALL_MODES = [
    ("full",      False, False),
    ("no-graph",  True,  False),
    ("no-vector", False, True),
    ("no-both",   True,  True),
]


def run_evaluation(
    golden_path: str,
    output_path: str,
    offline: bool = False,
    mode_filter: Optional[List[str]] = None,
) -> List[EvalResult]:
    """
    Main evaluation loop.

    Args:
        golden_path:  Path to golden_prs.yaml
        output_path:  Where to write the results CSV
        offline:      Use offline stub instead of a real LLM
        mode_filter:  Run only these mode names (None = all four)

    Returns:
        List of EvalResult objects (also written to CSV).
    """
    golden_prs = load_golden_prs(golden_path)
    active_modes = [
        (name, ng, nv)
        for (name, ng, nv) in ALL_MODES
        if mode_filter is None or name in mode_filter
    ]

    print(f"Evaluating {len(golden_prs)} golden PRs × {len(active_modes)} mode(s)…")
    results: List[EvalResult] = []

    for pr in golden_prs:
        for mode_name, no_graph, no_vector in active_modes:
            print(f"  [{pr.id}] mode={mode_name:<10}", end="", flush=True)

            contexts = build_contexts(pr, no_graph, no_vector)
            review   = (
                generate_review_offline(pr, contexts)
                if offline
                else generate_review_live(pr, contexts)
            )
            scores   = score_review(review, pr.expected_findings)

            # Ragas: only for full mode (to save API cost)
            rag_scores = _ragas_unavailable()
            if mode_name == "full" and not offline and contexts:
                ground_truth = "; ".join(
                    f"{ef.category} at {ef.file}:{ef.line or '?'} "
                    f"({', '.join(ef.must_mention)})"
                    for ef in pr.expected_findings
                )
                rag_scores = ragas_metrics(
                    question=pr.title,
                    answer=review.summary,
                    contexts=contexts,
                    ground_truth=ground_truth,
                )

            result = EvalResult(
                pr_id=pr.id,
                mode=mode_name,
                **rag_scores,
                **scores,
                findings_generated=len(review.findings),
                findings_expected=len(pr.expected_findings),
            )
            results.append(result)
            print(f"F1={result.f1:.3f}  TP={result.true_positives}  "
                  f"FP={result.false_positives}  FN={result.false_negatives}",
                  flush=True)

    _write_csv(results, output_path)
    _print_summary(results)
    return results


# ============================================================================
# Output helpers
# ============================================================================

def _write_csv(results: List[EvalResult], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not results:
        print("No results to write.")
        return
    fieldnames = list(asdict(results[0]).keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))
    print(f"\nResults → {path}")


def _print_summary(results: List[EvalResult]) -> None:
    by_mode: Dict[str, List[float]] = defaultdict(list)
    for r in results:
        by_mode[r.mode].append(r.f1)

    print("\n── Ablation Summary ─────────────────────────────────")
    print(f"{'Mode':<12}  {'Mean F1':>8}  {'PRs':>5}")
    print("─" * 38)
    full_mean: Optional[float] = None
    for name, _, _ in ALL_MODES:
        if name in by_mode:
            vals = by_mode[name]
            mean = sum(vals) / len(vals)
            print(f"{name:<12}  {mean:>8.4f}  {len(vals):>5}")
            if name == "full":
                full_mean = mean

    if full_mean is not None:
        print()
        for name, _, _ in ALL_MODES:
            if name != "full" and name in by_mode:
                vals = by_mode[name]
                delta = (sum(vals) / len(vals)) - full_mean
                print(f"  Δ {name:<12}: {delta:+.4f}  "
                      f"({'GraphRAG helps' if delta < -0.02 else 'no significant impact'})")
    print("─────────────────────────────────────────────────────")


# ============================================================================
# CI gate
# ============================================================================

def check_regression(
    results: List[EvalResult],
    baseline_path: str,
    threshold: float = 0.05,
) -> bool:
    """
    Return True (pass) if mean F1 for 'full' mode has NOT dropped more than
    `threshold` percentage points below the baseline stored in baseline_path.
    Returns True (pass) when no baseline file exists yet.
    """
    if not Path(baseline_path).exists():
        return True

    full_results = [r for r in results if r.mode == "full"]
    if not full_results:
        return True

    current_mean = sum(r.f1 for r in full_results) / len(full_results)

    with open(baseline_path) as f:
        reader = csv.DictReader(f)
        baseline_vals = [float(row["f1"]) for row in reader if row["mode"] == "full"]

    if not baseline_vals:
        return True

    baseline_mean = sum(baseline_vals) / len(baseline_vals)
    delta = current_mean - baseline_mean
    if delta < -threshold:
        print(
            f"\n❌ CI GATE FAILED: F1 dropped {abs(delta):.4f} "
            f"(threshold={threshold}). "
            f"Baseline={baseline_mean:.4f}, current={current_mean:.4f}",
            file=sys.stderr,
        )
        return False

    print(f"\n✅ CI gate passed: F1 Δ={delta:+.4f} (threshold={threshold})")
    return True


# ============================================================================
# Entry point
# ============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Graph Bug RAG evaluation harness (M9)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--golden",
        default=str(_HERE / "golden_prs.yaml"),
        help="Path to golden PRs YAML (default: evals/golden_prs.yaml)",
    )
    p.add_argument(
        "--output",
        default=str(_HERE / "results" / "latest.csv"),
        help="Output CSV path (default: evals/results/latest.csv)",
    )
    p.add_argument(
        "--baseline",
        default=str(_HERE / "results" / "baseline.csv"),
        help="Baseline CSV for CI regression gate",
    )
    p.add_argument(
        "--offline",
        action="store_true",
        help="Use offline stub (no LLM calls — for CI)",
    )
    p.add_argument(
        "--mode",
        action="append",
        dest="modes",
        choices=["full", "no-graph", "no-vector", "no-both"],
        help="Run only this mode (repeatable; default: all)",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="CI gate: max allowed F1 regression vs baseline (default: 0.05)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    results = run_evaluation(
        golden_path=args.golden,
        output_path=args.output,
        offline=args.offline,
        mode_filter=args.modes,
    )
    passed = check_regression(results, args.baseline, args.threshold)
    sys.exit(0 if passed else 1)
