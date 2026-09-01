"""Throughput/latency profiling, kept separate from accuracy runs.

Latency measured *during* a concurrent accuracy run is mostly queueing delay --
it describes your batch size, not the model. This module measures the two
things that are actually meaningful: single-stream latency (concurrency 1,
what a single interactive request feels like) and the throughput curve across
a concurrency sweep (where continuous batching saturates).

Uses inspect_ai's Model.generate() directly rather than eval() -- profiling
isn't a benchmark run and shouldn't go through the scoring/logging pipeline.
"""

from __future__ import annotations

import asyncio
import statistics
import time
from dataclasses import dataclass

from rich.console import Console
from rich.table import Table

from llm_bench.registry import Model, Registry

console = Console()

DEFAULT_PROMPT = (
    "Write a 300-word explanation of how continuous batching works in LLM "
    "inference servers, aimed at someone who understands transformers but not "
    "serving infrastructure."
)
DEFAULT_CONCURRENCIES = (1, 2, 4, 8, 16)
DEFAULT_SAMPLES_PER_LEVEL = 4


@dataclass
class ProfilePoint:
    model: str
    concurrency: int
    n: int
    wall_s: float
    mean_latency_s: float
    p50_latency_s: float
    total_output_tokens: int

    @property
    def aggregate_tok_s(self) -> float:
        return self.total_output_tokens / self.wall_s if self.wall_s else 0.0


async def _one_call(model, prompt: str, config) -> tuple[float, int]:
    start = time.monotonic()
    out = await model.generate(input=prompt, config=config)
    elapsed = time.monotonic() - start
    tokens = out.usage.output_tokens if out.usage else 0
    return elapsed, tokens


async def _sweep_level(model, prompt: str, config, concurrency: int, n: int) -> ProfilePoint:
    start = time.monotonic()
    results = await asyncio.gather(*[_one_call(model, prompt, config) for _ in range(n)])
    wall = time.monotonic() - start
    latencies = [r[0] for r in results]
    tokens = sum(r[1] for r in results)
    return ProfilePoint(
        model="",  # filled by caller
        concurrency=concurrency,
        n=n,
        wall_s=wall,
        mean_latency_s=statistics.mean(latencies),
        p50_latency_s=statistics.median(latencies),
        total_output_tokens=tokens,
    )


def profile_model(
    model: Model,
    prompt: str = DEFAULT_PROMPT,
    concurrencies: tuple[int, ...] = DEFAULT_CONCURRENCIES,
    samples_per_level: int = DEFAULT_SAMPLES_PER_LEVEL,
) -> list[ProfilePoint]:
    from inspect_ai.model import GenerateConfig, get_model

    m = get_model(model.spec)
    config = GenerateConfig(temperature=model.temperature, max_tokens=model.max_tokens)

    points: list[ProfilePoint] = []
    for c in concurrencies:
        n = max(c, samples_per_level)
        console.print(f"  concurrency={c} (n={n})...")
        point = asyncio.run(_sweep_level(m, prompt, config, c, n))
        point.model = model.name
        points.append(point)
    return points


def render(points: list[ProfilePoint]) -> Table:
    table = Table(title="throughput / latency profile")
    table.add_column("model")
    table.add_column("concurrency", justify="right")
    table.add_column("mean latency (s)", justify="right")
    table.add_column("p50 latency (s)", justify="right")
    table.add_column("aggregate tok/s", justify="right")

    for p in points:
        table.add_row(
            p.model, str(p.concurrency),
            f"{p.mean_latency_s:.2f}", f"{p.p50_latency_s:.2f}",
            f"{p.aggregate_tok_s:.1f}",
        )
    return table


def run_profile(model_names: list[str], **kwargs) -> None:
    registry = Registry.load()
    models = registry.resolve_models(model_names)
    all_points: list[ProfilePoint] = []
    for model in models:
        missing = model.missing_env()
        if missing:
            console.print(f"[red]{model.name}: missing {', '.join(missing)}, skipping[/red]")
            continue
        console.print(f"[cyan]profiling {model.name}[/cyan]")
        all_points.extend(profile_model(model, **kwargs))

    console.print(render(all_points))
    console.print(
        "[dim]concurrency=1 mean latency is the number to trust for 'how fast "
        "this feels'. The curve should rise then flatten -- if it's flat "
        "throughout, you're not saturating the server; if it never flattens, "
        "raise the concurrency sweep.[/dim]"
    )
