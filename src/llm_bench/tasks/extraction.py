"""Structured extraction eval -- the highest signal-per-effort custom task.

Fully deterministic: no judge model, no ambiguity, reproducible. Build your
golden set from real documents you have already processed, where you know the
right answer because you produced it. ~20-30 real cases beat a large synthetic set.
"""

from __future__ import annotations

import json
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, json_dataset
from inspect_ai.solver import generate, system_message

from llm_bench.registry import project_root
from llm_bench.tasks.scorers import field_f1, json_valid

SYSTEM = """You extract structured data from documents.

Respond with a single JSON object and nothing else. Use exactly the field names \
given in the schema. If a field is not present in the document, omit it rather \
than guessing -- a wrong value is worse than a missing one."""


def _record_to_sample(record: dict) -> Sample:
    schema = record.get("schema")
    prompt = record["input"]
    if schema:
        prompt = f"{prompt}\n\nSchema:\n{json.dumps(schema, indent=2)}"
    return Sample(
        input=prompt,
        target=json.dumps(record["target"]),
        id=record.get("id"),
        metadata=record.get("metadata", {}),
    )


@task
def extraction(dataset: str = "extraction.jsonl"):
    """Field-level F1 over your own documents.

    Args:
        dataset: filename under data/extraction/
    """
    path = project_root() / "data" / "extraction" / dataset
    if not path.exists():
        raise FileNotFoundError(
            f"no extraction dataset at {path}. See data/extraction/README.md "
            "for the expected format."
        )
    return Task(
        dataset=json_dataset(str(path), _record_to_sample),
        solver=[system_message(SYSTEM), generate()],
        # F1 is the headline; parse rate is tracked separately because a low
        # F1 from bad extraction and one from unparseable output are different
        # problems with different fixes.
        scorer=[field_f1(), json_valid()],
    )
