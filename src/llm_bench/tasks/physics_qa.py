"""Physics/math golden Q&A from your own domain.

Scored deterministically wherever the answer has a canonical form. The
model-graded path exists only for genuinely open-ended answers, and uses the
single pinned judge from models.yaml.
"""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, json_dataset
from inspect_ai.scorer import model_graded_qa
from inspect_ai.solver import generate, system_message

from llm_bench.registry import project_root
from llm_bench.tasks.scorers import normalized_match

SYSTEM = """You are answering a physics or mathematics question.

Reason step by step, then give your final answer on its own last line in the form:
ANSWER: <answer>

Give the answer in the units requested. Do not add commentary after the ANSWER line."""


def _record_to_sample(record: dict) -> Sample:
    return Sample(
        input=record["input"],
        target=str(record["target"]),
        id=record.get("id"),
        # `open_ended: true` routes an item to the judge instead of exact match
        metadata={"open_ended": record.get("open_ended", False),
                  **record.get("metadata", {})},
    )


@task
def physics_qa(dataset: str = "physics_qa.jsonl", open_ended: bool = False):
    """Golden Q&A over your domain.

    Args:
        dataset: filename under data/physics_qa/
        open_ended: use the pinned model judge instead of deterministic matching.
            Only for answers with no canonical form.
    """
    path = project_root() / "data" / "physics_qa" / dataset
    if not path.exists():
        raise FileNotFoundError(
            f"no physics_qa dataset at {path}. See data/physics_qa/README.md."
        )
    return Task(
        dataset=json_dataset(str(path), _record_to_sample),
        solver=[system_message(SYSTEM), generate()],
        scorer=model_graded_qa() if open_ended else normalized_match(),
    )
