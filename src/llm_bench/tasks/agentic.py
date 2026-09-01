"""Multi-step agentic eval -- the hardest custom task, build it last.

Scored on final state rather than trajectory: what matters is whether the task
got done, not whether the model took the route you imagined. Step count and
cost are recorded alongside, because agentic failure usually looks like
"succeeded, but burned 40 tool calls".
"""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.agent import react
from inspect_ai.dataset import Sample, json_dataset
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState
from inspect_ai.tool import bash, python
from inspect_ai.util import ExecResult, sandbox

from llm_bench.registry import project_root

VERIFY_TIMEOUT = 120
DEFAULT_MAX_STEPS = 20


def _record_to_sample(record: dict) -> Sample:
    return Sample(
        input=record["input"],
        # Shell/python snippet asserting the desired end state; non-zero exit = fail
        target=record["verify"],
        id=record.get("id"),
        files=record.get("files", {}),
        setup=record.get("setup"),
        metadata=record.get("metadata", {}),
    )


@scorer(metrics=[accuracy(), stderr()])
def final_state():
    """Run the verification script against the sandbox's end state."""

    async def score(state: TaskState, target: Target) -> Score:
        # Steps taken is a first-class quality signal, not just accuracy.
        steps = sum(
            1
            for m in state.messages
            if getattr(m, "role", None) == "assistant" and getattr(m, "tool_calls", None)
        )
        meta = {"steps": steps}

        try:
            result: ExecResult = await sandbox().exec(
                cmd=["bash", "-c", target.text], timeout=VERIFY_TIMEOUT
            )
        except TimeoutError:
            return Score(
                value=INCORRECT,
                explanation=f"verification timed out after {VERIFY_TIMEOUT}s",
                metadata={**meta, "error": "timeout"},
            )

        if result.success:
            return Score(
                value=CORRECT,
                explanation=f"final state verified in {steps} steps",
                metadata=meta,
            )
        return Score(
            value=INCORRECT,
            explanation=(result.stderr or result.stdout or "verification failed")[-600:],
            metadata={**meta, "returncode": result.returncode},
        )

    return score


@task
def agentic(dataset: str = "agentic.jsonl", max_steps: int = DEFAULT_MAX_STEPS):
    """Multi-step tasks from your workflows, scored on final state.

    Args:
        dataset: filename under data/agentic/
        max_steps: tool-call budget before the episode is cut off. A budget is
            required -- without one, a looping model runs up an unbounded bill.
    """
    path = project_root() / "data" / "agentic" / dataset
    if not path.exists():
        raise FileNotFoundError(
            f"no agentic dataset at {path}. See data/agentic/README.md."
        )
    return Task(
        dataset=json_dataset(str(path), _record_to_sample),
        solver=react(
            prompt=(
                "You are working in a Linux sandbox. Use the tools to complete "
                "the task. When you are done, say so -- your work is checked by "
                "inspecting the final state of the filesystem, not your summary."
            ),
            tools=[bash(timeout=VERIFY_TIMEOUT), python(timeout=VERIFY_TIMEOUT)],
            attempts=1,
        ),
        scorer=final_state(),
        sandbox="docker",
        message_limit=max_steps * 2,
    )
