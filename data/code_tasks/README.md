# Code / data-analysis tasks

One JSON object per line in `code_tasks.jsonl`. Pull these from real tasks in
your own projects — the closer to your actual work, the more the score means.

```json
{"id": "parse-spectrum", "input": "Write a function `peak_wavelength(spectrum: list[tuple[float,float]]) -> float` that returns the wavelength at maximum intensity.", "setup": "", "test": "assert abs(peak_wavelength([(400,0.1),(500,0.9),(600,0.2)]) - 500) < 1e-6"}
```

- `input`: the task prompt.
- `setup` (optional): code run before the model's solution (e.g. imports, fixtures).
- `test`: assertions run after the model's solution in the same process.
  Non-zero exit = fail.

Requires Docker — runs in `sandbox="docker"`. Scored by `executes_correctly`
in `src/llm_bench/tasks/code_tasks.py`.
