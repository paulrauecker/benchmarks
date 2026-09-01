# Agentic / multi-step tasks

One JSON object per line in `agentic.jsonl`. Hardest custom eval — build this
last, once extraction/code/physics_qa are working.

```json
{"id": "refactor-imports", "input": "In /work, find all Python files importing `old_module` and update them to import `new_module` instead.", "files": {"/work/a.py": "import old_module\n..."}, "verify": "grep -rL old_module /work/*.py && grep -rl new_module /work/*.py"}
```

- `input`: the task given to the react agent (tools: bash, python).
- `files` (optional): files seeded into the sandbox before the episode starts.
- `verify`: a bash script run against the sandbox's *final state* after the
  episode ends. Exit 0 = pass. Score on outcome, not trajectory — a model that
  reaches the right end state by an unexpected route should still pass.

Requires Docker. `max_steps` bounds the tool-call budget (default 20) — a
budget is mandatory, since a looping model without one runs up an unbounded
bill. Scored by `final_state` in `src/llm_bench/tasks/agentic.py`, which also
records step count as a first-class metric alongside pass/fail.
