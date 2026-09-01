"""Code / data-analysis eval scored by executing the model's output.

Deterministic: the model writes code, the sandbox runs it against your
assertions, pass/fail. Pull these from real tasks in your own projects -- the
more they resemble your actual work, the more the score means.

Requires Docker (sandbox="docker"). Model-generated code is never run on the host.
"""

from __future__ import annotations

import re

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, json_dataset
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState, generate, system_message
from inspect_ai.util import ExecResult, sandbox

from llm_bench.registry import project_root

SYSTEM = """You write Python to solve the given task.

Return only the code, in a single ```python fenced block. Do not include \
explanation, tests, or example usage -- your code will be executed directly \
against a hidden test suite. Define exactly the names the task asks for."""

VERIFY_TIMEOUT = 120


def _extract_code(completion: str) -> str:
    """Take the fenced block, or the whole response if the model omitted fences."""
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", completion, re.DOTALL)
    if blocks:
        # Last block: models often show a sketch before the final answer.
        return blocks[-1]
    return completion


def _record_to_sample(record: dict) -> Sample:
    return Sample(
        input=record["input"],
        target=record["test"],           # assertion block, run after the solution
        id=record.get("id"),
        metadata={"setup": record.get("setup", ""), **record.get("metadata", {})},
    )


@scorer(metrics=[accuracy(), stderr()])
def executes_correctly():
    """Run solution + assertions in the sandbox; pass only on a clean exit."""

    async def score(state: TaskState, target: Target) -> Score:
        code = _extract_code(state.output.completion)
        setup = state.metadata.get("setup", "")
        program = "\n\n".join(p for p in (setup, code, target.text) if p)

        try:
            result: ExecResult = await sandbox().exec(
                cmd=["python", "-c", program],
                timeout=VERIFY_TIMEOUT,
            )
        except TimeoutError:
            return Score(
                value=INCORRECT,
                answer=code[:200],
                explanation=f"timed out after {VERIFY_TIMEOUT}s",
                metadata={"error": "timeout"},
            )

        if result.success:
            return Score(value=CORRECT, answer=code[:200])
        return Score(
            value=INCORRECT,
            answer=code[:200],
            explanation=(result.stderr or result.stdout or "non-zero exit")[-600:],
            metadata={"returncode": result.returncode},
        )

    return score


@task
def code_tasks(dataset: str = "code_tasks.jsonl"):
    """Assertion-scored coding tasks from your workflows.

    Args:
        dataset: filename under data/code_tasks/
    """
    path = project_root() / "data" / "code_tasks" / dataset
    if not path.exists():
        raise FileNotFoundError(
            f"no code_tasks dataset at {path}. See data/code_tasks/README.md."
        )
    return Task(
        dataset=json_dataset(str(path), _record_to_sample),
        solver=[system_message(SYSTEM), generate()],
        scorer=executes_correctly(),
        sandbox="docker",
    )
