"""Turn a log directory into one comparison table.

Deliberately shows stderr next to every score and flags overlapping confidence
intervals -- the point is to stop you reading a ranking into noise. GPQA-Diamond
at n=198 has a ~7pp 95% CI; most "wins" between similarly-sized models are not
real. See the plan's "Small benchmarks have brutal error bars" section.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()


@dataclass
class Cell:
    model: str
    task: str
    score: float | None
    stderr: float | None
    n_scored: int
    n_total: int
    cost: float
    duration_s: float
    confidence: float | None = None  # mean logprob-derived confidence, if available
    ece: float | None = None
    # cost, duration and sample counts belong to the eval run, not to the scorer.
    # A task with two scorers yields two cells carrying the same run-level
    # numbers; only the one flagged here may be summed, or every total is
    # multiplied by the scorer count.
    owns_run_totals: bool = True

    @property
    def error_rate(self) -> float:
        if self.n_total == 0:
            return 0.0
        return 1 - (self.n_scored / self.n_total)

    @property
    def ci95(self) -> tuple[float, float] | None:
        if self.score is None or self.stderr is None:
            return None
        return (self.score - 1.96 * self.stderr, self.score + 1.96 * self.stderr)

    def overlaps(self, other: "Cell") -> bool:
        a, b = self.ci95, other.ci95
        if a is None or b is None:
            return False
        return not (a[1] < b[0] or b[1] < a[0])


def _headline_metric(score_entry, headline=None) -> tuple[float | None, float | None]:
    """Pick the primary metric and its stderr from an EvalScore.

    Benchmarks namespace their metrics (IFEval's prompt_strict_acc /
    prompt_strict_stderr, final_acc / final_stderr, ...), so there's no
    metric literally called "accuracy" or "stderr" to grab. inspect_ai
    already resolves which metric is the intended headline per task
    (EvalResults.headline) -- prefer that over guessing from key order,
    which is fragile and was previously silently picking whichever metric
    happened to be inserted first.
    """
    metrics = score_entry.metrics
    primary_key = None

    if (
        headline is not None
        and headline.scorer == score_entry.scorer
        and headline.metric in metrics
    ):
        primary_key = headline.metric
    elif "accuracy" in metrics:
        primary_key = "accuracy"
    elif "mean" in metrics:
        primary_key = "mean"
    elif metrics:
        primary_key = next(iter(metrics))

    if primary_key is None:
        return None, None

    primary = metrics[primary_key].value

    # Find the matching stderr: same key with _acc->_stderr, or "<key>_stderr",
    # falling back to a bare "stderr" entry. Guard against a candidate that's
    # just primary_key unchanged (e.g. "accuracy".replace("_acc","_stderr") is
    # a no-op since there's no underscore) -- that would otherwise match the
    # primary metric itself and report its own value as its stderr.
    err = None
    for candidate in (
        primary_key.replace("_acc", "_stderr"),
        f"{primary_key}_stderr",
        "stderr",
    ):
        if candidate != primary_key and candidate in metrics:
            err = metrics[candidate].value
            break

    return primary, err


def _duration_seconds(started: str, completed: str) -> float:
    if not started or not completed:
        return 0.0
    try:
        return (
            datetime.fromisoformat(completed) - datetime.fromisoformat(started)
        ).total_seconds()
    except ValueError:
        return 0.0


def _started_key(log, name: str) -> tuple[str, str]:
    """Sort key for "which run is newer".

    `stats.started_at` is the truth; the log's name -- which inspect prefixes
    with the run's timestamp -- breaks ties and stands in when a log lacks the
    stat. Both are ISO-ish and sort lexicographically, which is all that's
    needed to order runs inside one directory.
    """
    return (log.stats.started_at or "", name)


@dataclass
class RunRef:
    """A superseded run, kept only so the reader can be told it was dropped."""

    model: str
    task: str
    name: str
    started_at: str


@dataclass
class Collection:
    cells: list[Cell]
    superseded: list[RunRef]


def collect(log_dir: Path) -> Collection:
    """Read a log dir, keeping only the newest successful run per model+task.

    Re-running a suite into an existing log dir leaves several successful logs
    for the same model and task. Averaging or summing across them would
    double-count cost and plot the same model twice, so the newest wins and the
    older ones are reported as superseded rather than silently dropped.
    """
    from inspect_ai.log import list_eval_logs, read_eval_log

    newest: dict[tuple[str, str], tuple[tuple[str, str], object, str]] = {}
    superseded: list[RunRef] = []

    for info in list_eval_logs(str(log_dir)):
        log = read_eval_log(info, header_only=True)
        if log.status != "success" or not log.results:
            console.print(
                f"[yellow]skipping {info.name}: status={log.status}[/yellow]"
            )
            continue

        key = (log.eval.model, log.eval.task)
        started = _started_key(log, info.name)
        previous = newest.get(key)
        if previous is None or started > previous[0]:
            if previous is not None:
                superseded.append(
                    RunRef(key[0], key[1], previous[2], previous[0][0])
                )
            newest[key] = (started, log, info.name)
        else:
            superseded.append(RunRef(key[0], key[1], info.name, started[0]))

    cells: list[Cell] = []
    for (model, task), (_started, log, _name) in newest.items():
        usage = log.stats.model_usage.get(model) if log.stats.model_usage else None
        cost = usage.total_cost if (usage and usage.total_cost) else 0.0
        duration = _duration_seconds(log.stats.started_at, log.stats.completed_at)

        for index, score_entry in enumerate(log.results.scores):
            value, err = _headline_metric(score_entry, log.results.headline)
            cells.append(
                Cell(
                    model=model,
                    task=f"{task}/{score_entry.name}",
                    score=value,
                    stderr=err,
                    n_scored=log.results.completed_samples,
                    n_total=log.results.total_samples,
                    cost=cost,
                    duration_s=duration,
                    owns_run_totals=index == 0,
                )
            )
    return Collection(cells=cells, superseded=superseded)


def render_table(cells: list[Cell]) -> Table:
    by_model: dict[str, dict[str, Cell]] = defaultdict(dict)
    tasks: list[str] = []
    for c in cells:
        if c.task not in tasks:
            tasks.append(c.task)
        by_model[c.model][c.task] = c

    table = Table(title="llm-bench results", show_lines=False)
    table.add_column("model", style="bold")
    for t in tasks:
        table.add_column(t, justify="right")
    table.add_column("cost $", justify="right")
    table.add_column("errs", justify="right")

    # For overlap flagging: best cell per task across models.
    best: dict[str, Cell] = {}
    for t in tasks:
        col = [by_model[m][t] for m in by_model if t in by_model[m] and by_model[m][t].score is not None]
        if col:
            best[t] = max(col, key=lambda c: c.score)

    for model, row in by_model.items():
        cells_out = []
        total_cost = 0.0
        total_err = 0
        for t in tasks:
            c = row.get(t)
            if c is None or c.score is None:
                cells_out.append("-")
                continue
            if c.owns_run_totals:
                total_cost += c.cost
                total_err += int(round(c.error_rate * c.n_total))
            txt = f"{c.score:.3f}"
            if c.stderr is not None:
                txt += f"±{c.stderr:.3f}"
            b = best.get(t)
            if b is not None and b is not c and not c.overlaps(b):
                pass  # clearly behind the leader; no marker needed
            elif b is not None and b is c:
                txt = f"[green]{txt}[/green]"
            elif b is not None and c.overlaps(b):
                txt = f"{txt}[dim]~[/dim]"  # statistically tied with the leader
            cells_out.append(txt)
        cells_out.append(f"{total_cost:.2f}")
        cells_out.append(str(total_err) if total_err else "0")
        table.add_row(model, *cells_out)

    return table


def report(log_dir: Path) -> None:
    collection = collect(log_dir)
    if not collection.cells:
        console.print(f"[red]no completed eval logs found under {log_dir}[/red]")
        return
    table = render_table(collection.cells)
    console.print(table)
    for ref in collection.superseded:
        console.print(
            f"[yellow]superseded[/yellow] {ref.name} -- an older run of "
            f"{ref.task} on {ref.model}; showing the newest only"
        )
    console.print(
        "[dim]~ marks scores within the leader's 95% CI (not distinguishable). "
        "'errs' counts unscored samples (timeouts, parse failures, refusals).[/dim]"
    )
