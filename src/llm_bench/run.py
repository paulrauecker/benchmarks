"""Suite runner.

Drives inspect_ai's `eval()` over the cartesian product of models and tasks,
applying the registry's per-model settings so every model is treated identically.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console

from llm_bench.registry import Model, Registry, TaskSpec, project_root

console = Console()

# Spend guards. A per-sample cost ceiling is a circuit breaker against a model
# that loops or emits unbounded reasoning -- without it, one bad task can cost
# more than the entire rest of the suite.
DEFAULT_COST_LIMIT = 1.00
DEFAULT_TOKEN_LIMIT = 100_000

# Tasks that pull a gated HuggingFace dataset. Confirmed via HfApi.dataset_info:
# gpqa_diamond -> Idavidrein/gpqa (gated: auto); the rest of the core/math/code
# suite (ifeval, mmlu_pro, math, aime2025, humaneval, mbpp) is not gated. An
# unauthenticated load doesn't fail fast -- it hangs retrying before eventually
# raising DatasetNotFoundError, so this is checked explicitly up front.
GATED_HF_TASKS = {"gpqa_diamond"}


def preflight(
    models: list[Model], registry: Registry, tasks: list[TaskSpec] | None = None
) -> list[str]:
    """Check env vars and judge config before spending anything."""
    problems = []
    if tasks and any(t.name in GATED_HF_TASKS for t in tasks):
        if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
            gated = [t.name for t in tasks if t.name in GATED_HF_TASKS]
            problems.append(
                f"{', '.join(gated)}: requires a HuggingFace account that has "
                f"accepted the dataset's terms, plus HF_TOKEN set. Without it, "
                f"the load doesn't fail fast -- it hangs retrying for several "
                f"minutes before erroring. Accept terms at "
                f"https://huggingface.co/datasets/Idavidrein/gpqa and set "
                f"HF_TOKEN in .env."
            )
    for m in models:
        missing = m.missing_env()
        if missing:
            problems.append(
                f"{m.name}: missing env var(s) {', '.join(missing)} "
                f"(provider {m.provider!r})"
            )
        try:
            registry.check_judge(m)
        except ValueError as e:
            problems.append(str(e))

        # A llama.cpp slot budget that cannot fit max_tokens will truncate
        # requests mid-run and surface as wrong answers, not as errors.
        if m.server:
            parallel = m.server.get("parallel")
            ctx = m.server.get("ctx_size")
            if parallel and ctx:
                per_slot = ctx // parallel
                if per_slot < m.max_tokens:
                    problems.append(
                        f"{m.name}: llama.cpp slot context is {per_slot} tokens "
                        f"(ctx_size {ctx} / parallel {parallel}) but max_tokens is "
                        f"{m.max_tokens}. Requests will truncate silently. "
                        f"Raise --ctx-size to at least {m.max_tokens * parallel}."
                    )
            if parallel and m.max_connections > parallel:
                problems.append(
                    f"{m.name}: max_connections ({m.max_connections}) exceeds "
                    f"llama-server --parallel ({parallel}); requests will thrash "
                    f"slots. Lower max_connections to {parallel}."
                )
    return problems


def run_suite(
    suite: str,
    model_names: list[str],
    limit: int | None = None,
    epochs: int | None = None,
    log_dir: Path | None = None,
    dry_run: bool = False,
    force: bool = False,
) -> Path:
    """Run every task in `suite` against every model in `model_names`."""
    registry = Registry.load()
    models = registry.resolve_models(model_names)
    tasks = registry.resolve_suite(suite)

    problems = preflight(models, registry, tasks)
    if problems:
        for p in problems:
            console.print(f"[red]preflight:[/red] {p}")
        if not force:
            raise SystemExit(
                "Refusing to run. Fix the above, or pass --force to override."
            )
        console.print("[yellow]--force given; continuing despite preflight errors[/]")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = log_dir or project_root() / "logs" / f"{stamp}-{suite}"

    def _eff_display(t):
        eff = TaskSpec(t.name, t.path, limit if limit is not None else t.limit,
                        epochs if epochs is not None else t.epochs)
        return eff.display()

    console.print(f"[bold]suite[/bold]  {suite}: {', '.join(_eff_display(t) for t in tasks)}")
    console.print(f"[bold]models[/bold] {', '.join(m.name for m in models)}")
    console.print(f"[bold]logs[/bold]   {log_dir}")

    if dry_run:
        console.print("[yellow]dry run; nothing executed[/]")
        return log_dir

    from inspect_ai import eval as inspect_eval

    log_dir.mkdir(parents=True, exist_ok=True)

    for model in models:
        for spec in tasks:
            _run_one(inspect_eval, model, spec, registry, limit, epochs, log_dir)

    console.print(f"\n[green]done[/green]  llm-bench report {log_dir}")
    return log_dir


def resume_incomplete(log_dir: Path) -> None:
    """Resume every non-successful run under `log_dir`.

    `retry_on_error` already absorbs transient per-sample blips inside a
    running eval. This covers the harder case: the whole `llm-bench run`
    process died mid-suite (retries exhausted, process killed, connectivity
    gone long enough to give up). inspect's log format records per-sample
    completion, and `eval_retry` resumes a task from its log, re-running
    only the samples that never finished -- so nothing already-completed
    is redone or re-billed.
    """
    from inspect_ai import eval_retry
    from inspect_ai.log import list_eval_logs, read_eval_log

    infos = list_eval_logs(str(log_dir))
    incomplete = []
    for info in infos:
        log = read_eval_log(info, header_only=True)
        if log.status != "success":
            incomplete.append(info)

    if not incomplete:
        console.print(f"[green]nothing to resume[/green] -- all runs under {log_dir} succeeded")
        return

    console.print(f"[bold]resuming[/bold] {len(incomplete)} incomplete run(s):")
    for info in incomplete:
        console.print(f"  {info.name}")

    eval_retry(incomplete, log_dir=str(log_dir))


def _run_one(
    inspect_eval: Any,
    model: Model,
    spec: TaskSpec,
    registry: Registry,
    limit: int | None,
    epochs: int | None,
    log_dir: Path,
) -> None:
    """One (model, task) cell. Errors are contained so one failure doesn't
    abandon a long multi-model run."""
    eff_limit = limit if limit is not None else spec.limit
    eff_epochs = epochs if epochs is not None else spec.epochs

    console.print(f"\n[cyan]==> {model.name} / {spec.display()}[/cyan]")

    kwargs: dict[str, Any] = {
        "tasks": spec.path,
        "model": model.spec,
        "log_dir": str(log_dir),
        # Score rather than abort on a sample error, so one bad item doesn't
        # discard an otherwise complete run.
        "score_on_error": True,
        # OpenRouter rate-limits; retrying is the correct response, not
        # permanently lowering concurrency.
        "retry_on_error": 3,
        # Spend guards -- see module docstring.
        "token_limit": DEFAULT_TOKEN_LIMIT,
        "cost_limit": DEFAULT_COST_LIMIT,
        # Generation settings arrive as flat kwargs (Unpack[GenerateConfigArgs]),
        # not as a config object.
        **model.generate_config(),
    }
    if eff_limit is not None:
        kwargs["limit"] = eff_limit
    if eff_epochs is not None:
        kwargs["epochs"] = eff_epochs
    if model.cost is not None:
        registry.register_model_cost(model)
    # Only wire the grader role for tasks that actually use model-graded
    # scoring, and only when its credentials are present. Wiring it
    # unconditionally for every task (a) initialises a client the task
    # doesn't need and (b) makes `cost_limit` demand pricing data for the
    # judge even on runs that never call it, since inspect can't tell an
    # unused model_role apart from a used one.
    if spec.name in ("physics_qa", "agentic") and registry.judge:
        if registry.judge_available():
            kwargs["model_roles"] = {"grader": registry.judge}
            judge = registry.judge_model()
            if judge is not None:
                registry.register_model_cost(judge)
        else:
            console.print(
                f"[yellow]warning:[/yellow] judge ({registry.judge}) has no "
                f"credentials in this environment; model-graded scoring in "
                f"{spec.name!r} will fail if it hits an open-ended item"
            )

    try:
        inspect_eval(**kwargs)
    except Exception as e:  # noqa: BLE001 - keep the rest of the matrix running
        console.print(f"[red]failed[/red] {model.name}/{spec.name}: {e}")
