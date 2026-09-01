"""Deterministic scorers shared by the custom tasks.

Deterministic scoring is preferred wherever it is achievable -- it is
reproducible, free, and immune to the self-preference bias that model-graded
scoring exhibits. Model grading is reserved for genuinely open-ended answers.
"""

from __future__ import annotations

import json
import re
from typing import Any

from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Target,
    accuracy,
    mean,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState


def extract_json(text: str) -> Any | None:
    """Pull a JSON object out of a model response.

    Models wrap JSON in prose or fences even when told not to. Failing to
    handle that would measure formatting compliance instead of extraction
    quality -- so we strip fences, then fall back to brace matching.
    """
    if not text:
        return None

    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    candidates = [fenced.group(1)] if fenced else []
    candidates.append(text)

    for cand in candidates:
        cand = cand.strip()
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            pass
        # Brace matching for JSON embedded in surrounding prose.
        start = cand.find("{")
        while start != -1:
            depth, in_str, esc = 0, False, False
            for i in range(start, len(cand)):
                ch = cand[i]
                if esc:
                    esc = False
                    continue
                if ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = not in_str
                elif not in_str:
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            try:
                                return json.loads(cand[start : i + 1])
                            except json.JSONDecodeError:
                                break
            start = cand.find("{", start + 1)
    return None


def _normalize_scalar(v: Any) -> str:
    """Compare values without punishing trivial formatting differences."""
    if v is None:
        return ""
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, (int, float)):
        # 3 and 3.0 and "3" should match
        f = float(v)
        return str(int(f)) if f.is_integer() else f"{f:g}"
    s = str(v).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _flatten(obj: Any, prefix: str = "") -> dict[str, str]:
    """Flatten nested JSON to dotted leaf paths for field-level comparison."""
    out: dict[str, str] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list):
        # Order-insensitive: lists in extraction targets are usually sets.
        for item in sorted(_normalize_scalar(x) if not isinstance(x, (dict, list))
                           else json.dumps(x, sort_keys=True) for x in obj):
            out[f"{prefix}[]={item}"] = item
    else:
        out[prefix] = _normalize_scalar(obj)
    return out


@scorer(metrics=[mean(), stderr()])
def field_f1():
    """Field-level F1 for structured extraction.

    Scores per-key against ground truth rather than requiring an exact document
    match, so a model that gets 9 of 10 fields right scores 0.9 rather than 0.
    This is the failure mode that matters in production: schema drift and
    hallucinated fields, not whitespace.
    """

    async def score(state: TaskState, target: Target) -> Score:
        parsed = extract_json(state.output.completion)
        if parsed is None:
            return Score(
                value=0.0,
                answer=state.output.completion[:200],
                explanation="response contained no parseable JSON",
                metadata={"json_parsed": False},
            )

        expected = _flatten(json.loads(target.text))
        actual = _flatten(parsed)

        # Empty values count as absent, not as wrong-valued.
        expected = {k: v for k, v in expected.items() if v != ""}
        actual = {k: v for k, v in actual.items() if v != ""}

        tp = sum(1 for k, v in expected.items() if actual.get(k) == v)
        fp = len(actual) - sum(1 for k in actual if k in expected and actual[k] == expected[k])
        fn = len(expected) - tp

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        missing = sorted(k for k, v in expected.items() if actual.get(k) != v)
        spurious = sorted(k for k in actual if k not in expected)

        return Score(
            value=f1,
            answer=json.dumps(parsed)[:200],
            explanation=(
                f"P={precision:.2f} R={recall:.2f} F1={f1:.2f}; "
                f"wrong/missing={missing[:5]}; hallucinated={spurious[:5]}"
            ),
            metadata={
                "json_parsed": True,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "n_missing": len(missing),
                "n_hallucinated": len(spurious),
            },
        )

    return score


@scorer(metrics=[accuracy(), stderr()])
def json_valid():
    """Bare parse rate -- tracked separately from F1.

    A model can score poorly on F1 either because it extracts badly or because
    it cannot emit JSON at all. Those call for different responses, so they are
    measured separately.
    """

    async def score(state: TaskState, target: Target) -> Score:
        ok = extract_json(state.output.completion) is not None
        return Score(value=CORRECT if ok else INCORRECT)

    return score


_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?")


@scorer(metrics=[accuracy(), stderr()])
def normalized_match(numeric_tolerance: float = 1e-6):
    """Exact match after normalisation, with numeric tolerance.

    For physics/math answers that are numbers or short expressions. Prefer this
    over a model judge wherever the answer has a canonical form -- it removes
    both the cost and the bias of an LLM grader.
    """

    async def score(state: TaskState, target: Target) -> Score:
        completion = state.output.completion or ""
        expected = target.text.strip()

        # Prefer a final-answer marker if the prompt asked for one.
        marker = re.search(
            r"(?:ANSWER|Answer)\s*[:=]\s*(.+?)(?:\n|$)", completion
        )
        candidate = marker.group(1).strip() if marker else completion.strip()

        if _normalize_scalar(candidate) == _normalize_scalar(expected):
            return Score(value=CORRECT, answer=candidate[:200])

        # Numeric comparison with tolerance, using the last number in the
        # response when no explicit marker was given.
        exp_nums = _NUM_RE.findall(expected.replace(",", ""))
        got_nums = _NUM_RE.findall(candidate.replace(",", ""))
        if exp_nums and got_nums:
            try:
                e, g = float(exp_nums[-1]), float(got_nums[-1])
                scale = max(abs(e), 1.0)
                if abs(e - g) <= numeric_tolerance * scale:
                    return Score(value=CORRECT, answer=candidate[:200])
            except ValueError:
                pass

        return Score(
            value=INCORRECT,
            answer=candidate[:200],
            explanation=f"expected {expected!r}",
        )

    return score
