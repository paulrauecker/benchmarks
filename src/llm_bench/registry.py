"""Model and suite registry.

Loads models.yaml / suites.yaml and renders them into the arguments inspect_ai
expects. Centralising generation settings here is what guarantees a self-hosted
model and an OpenRouter model get identical treatment -- the most common way a
homemade benchmark lies is a template or sampling difference between those paths.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Provider prefixes and the env vars they need. Used to fail fast with a useful
# message instead of surfacing an auth error 200 samples into a paid run.
PROVIDER_ENV = {
    "openrouter": ["OPENROUTER_API_KEY"],
    "openai-api": [],  # checked dynamically: <PROVIDER>_BASE_URL
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass
class Model:
    name: str
    spec: str
    temperature: float
    max_tokens: int
    max_connections: int
    logprobs: bool
    cost: dict[str, float] | None = None
    server: dict[str, Any] | None = None
    reasoning_effort: str | None = None
    top_p: float | None = None
    top_k: int | None = None
    presence_penalty: float | None = None
    timeout: float | None = None

    @property
    def provider(self) -> str:
        return self.spec.split("/", 1)[0]

    @property
    def is_self_hosted(self) -> bool:
        return self.provider in ("openai-api", "vllm", "ollama", "llama-cpp-python")

    def required_env(self) -> list[str]:
        """Env vars that must be set for this model to be reachable."""
        if self.provider == "openai-api":
            # openai-api/<provider>/<model> -> <PROVIDER>_BASE_URL / _API_KEY
            parts = self.spec.split("/")
            if len(parts) < 3:
                raise ValueError(
                    f"model {self.name!r}: openai-api specs must be "
                    f"'openai-api/<provider>/<model>', got {self.spec!r}"
                )
            stem = parts[1].upper().replace("-", "_")
            return [f"{stem}_BASE_URL", f"{stem}_API_KEY"]
        return PROVIDER_ENV.get(self.provider, [])

    def missing_env(self) -> list[str]:
        return [v for v in self.required_env() if not os.environ.get(v)]

    def generate_config(self) -> dict[str, Any]:
        """Kwargs for inspect_ai GenerateConfig."""
        cfg: dict[str, Any] = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "max_connections": self.max_connections,
        }
        if self.logprobs:
            # Completion-token logprobs only. `prompt_logprobs` (needed for true
            # loglikelihood scoring) is a vLLM extension that llama.cpp and
            # OpenRouter do not provide, so we never request it -- accuracy is
            # always scored generatively. These feed confidence/ECE columns.
            cfg["logprobs"] = True
            cfg["top_logprobs"] = 5
        if self.reasoning_effort:
            # Left unset, a reasoning model doesn't fall back to some sane
            # default -- it reasons unconstrained. Confirmed live: an
            # llama.cpp Qwen3 box with no reasoning_effort set spent most of
            # its max_tokens budget on reasoning_content on nearly every
            # sample, ~8x slower than a comparable capped run. Also confirmed
            # OpenRouter accepts this field even for models that don't list
            # it in supported_parameters (silently no-ops rather than 400s).
            cfg["reasoning_effort"] = self.reasoning_effort
        # Defaults to greedy (temperature 0) for cross-model reproducibility --
        # the usual eval-harness convention. Some model cards (e.g. Qwen3.8
        # Flash-Next, Qwen3.8-27B) explicitly recommend against greedy
        # decoding for their reasoning-tuned variants and publish their own
        # sampling params instead; those models override temperature/top_p/
        # top_k/presence_penalty here rather than through this being unset.
        if self.top_p is not None:
            cfg["top_p"] = self.top_p
        if self.top_k is not None:
            cfg["top_k"] = self.top_k
        if self.presence_penalty is not None:
            cfg["presence_penalty"] = self.presence_penalty
        if self.timeout is not None:
            # inspect_ai's per-request default is 600s. Confirmed live this
            # is too short for a slow self-hosted reasoning model: at ~8
            # tok/s decode (observed on the ServerS box under load) a
            # max_tokens: 8192 generation can take ~1024s worst case, so the
            # client cancels mid-generation and retries from scratch --
            # losing all progress and compounding into a retry storm rather
            # than just running long once.
            cfg["timeout"] = self.timeout
        return cfg


@dataclass
class TaskSpec:
    """One task in a suite, with any per-task overrides."""

    name: str
    path: str
    limit: int | None = None
    epochs: int | None = None
    # Task-constructor args (inspect's `task_args`). Used to turn OFF a task's
    # own unseeded dataset shuffle so that the seeded one in run.py governs
    # item selection -- see DEFAULT_SAMPLE_SHUFFLE_SEED there. Only some tasks
    # accept `shuffle`; passing it to one that doesn't is an error, hence
    # per-task rather than global.
    args: dict[str, Any] = field(default_factory=dict)

    def display(self) -> str:
        bits = [self.name]
        if self.limit:
            bits.append(f"limit={self.limit}")
        if self.epochs:
            bits.append(f"epochs={self.epochs}")
        return " ".join(bits)


@dataclass
class Registry:
    models: dict[str, Model]
    judge: str
    judge_cost: dict[str, float] | None
    suites: dict[str, dict[str, Any]]
    task_paths: dict[str, str]
    _suite_defaults: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load(
        cls, models_path: Path | None = None, suites_path: Path | None = None
    ) -> Registry:
        root = project_root()
        models_raw = yaml.safe_load((models_path or root / "models.yaml").read_text())
        suites_raw = yaml.safe_load((suites_path or root / "suites.yaml").read_text())

        defaults = models_raw.get("defaults", {})
        models: dict[str, Model] = {}
        for name, cfg in (models_raw.get("models") or {}).items():
            if "spec" not in cfg:
                raise ValueError(f"model {name!r} has no 'spec'")
            models[name] = Model(
                name=name,
                spec=cfg["spec"],
                temperature=cfg.get("temperature", defaults.get("temperature", 0.0)),
                max_tokens=cfg.get("max_tokens", defaults.get("max_tokens", 4096)),
                max_connections=cfg.get(
                    "max_connections", defaults.get("max_connections", 8)
                ),
                logprobs=cfg.get("logprobs", defaults.get("logprobs", False)),
                cost=cfg.get("cost"),
                server=cfg.get("server"),
                reasoning_effort=cfg.get(
                    "reasoning_effort", defaults.get("reasoning_effort")
                ),
                top_p=cfg.get("top_p", defaults.get("top_p")),
                top_k=cfg.get("top_k", defaults.get("top_k")),
                presence_penalty=cfg.get(
                    "presence_penalty", defaults.get("presence_penalty")
                ),
                timeout=cfg.get("timeout", defaults.get("timeout")),
            )

        return cls(
            models=models,
            judge=models_raw.get("judge", ""),
            judge_cost=models_raw.get("judge_cost"),
            suites=suites_raw.get("suites", {}),
            task_paths=suites_raw.get("registry", {}),
        )

    def resolve_models(self, names: list[str]) -> list[Model]:
        unknown = [n for n in names if n not in self.models]
        if unknown:
            raise KeyError(
                f"unknown model(s): {', '.join(unknown)}. "
                f"Available: {', '.join(sorted(self.models))}"
            )
        return [self.models[n] for n in names]

    def resolve_suite(self, suite_name: str, _seen: set[str] | None = None) -> list[TaskSpec]:
        """Expand a suite into concrete TaskSpecs, following `includes`."""
        _seen = _seen or set()
        if suite_name in _seen:
            raise ValueError(f"circular suite include involving {suite_name!r}")
        _seen.add(suite_name)

        if suite_name not in self.suites:
            raise KeyError(
                f"unknown suite {suite_name!r}. "
                f"Available: {', '.join(sorted(self.suites))}"
            )
        suite = self.suites[suite_name]
        suite_limit = suite.get("limit")
        specs: list[TaskSpec] = []

        for included in suite.get("includes", []):
            specs.extend(self.resolve_suite(included, _seen))

        for entry in suite.get("tasks", []):
            if isinstance(entry, str):
                entry = {"task": entry}
            name = entry["task"]
            if name not in self.task_paths:
                raise KeyError(
                    f"task {name!r} (in suite {suite_name!r}) is not in the "
                    f"suites.yaml `registry:` map"
                )
            specs.append(
                TaskSpec(
                    name=name,
                    path=self.task_paths[name],
                    limit=entry.get("limit", suite_limit),
                    epochs=entry.get("epochs"),
                    args=entry.get("args") or {},
                )
            )

        # de-dupe by name, keeping the first (tighter) spec
        seen_names: set[str] = set()
        deduped = []
        for s in specs:
            if s.name not in seen_names:
                seen_names.add(s.name)
                deduped.append(s)
        return deduped

    def judge_model(self) -> Model | None:
        """Wrap the configured judge spec as a Model so its env can be checked."""
        if not self.judge:
            return None
        return Model(
            name="judge", spec=self.judge, temperature=0.0, max_tokens=4096,
            max_connections=4, logprobs=False, cost=self.judge_cost,
        )

    def judge_available(self) -> bool:
        j = self.judge_model()
        return j is not None and not j.missing_env()

    def register_model_cost(self, model: Model) -> None:
        """Register pricing for a custom model name with inspect_ai.

        inspect's --model-cost-config / set_model_cost() only OVERRIDES cost
        on models already present in inspect's built-in pricing database --
        it raises "Model '<name>' not found" for anything outside it, which
        includes every spec in this registry (self-hosted paths and OpenRouter
        model strings alike; verified none of them resolve via
        inspect_ai.model.get_model_info). set_model_info() is the primitive
        that registers a brand-new model, so that's what we use here instead.

        Cache read/write pricing is not modeled (defaults to $0) since it
        varies by provider and most of this registry's models are self-hosted
        anyway; total_cost will underestimate slightly for cached OpenRouter
        calls.
        """
        if model.cost is None:
            return
        from inspect_ai.model import ModelCost, ModelInfo, set_model_info

        set_model_info(
            model.spec,
            ModelInfo(
                cost=ModelCost(
                    input=float(model.cost.get("input", 0.0)),
                    output=float(model.cost.get("output", 0.0)),
                    input_cache_write=float(model.cost.get("input_cache_write", 0.0)),
                    input_cache_read=float(model.cost.get("input_cache_read", 0.0)),
                )
            ),
        )

    def check_judge(self, model: Model) -> None:
        """Refuse to let a model grade itself."""
        if not self.judge:
            return
        judge_stem = self.judge.split("/")[-1].lower()
        model_stem = model.spec.split("/")[-1].lower()
        if judge_stem == model_stem:
            raise ValueError(
                f"model {model.name!r} would be graded by itself ({self.judge}). "
                "Pick a different judge in models.yaml."
            )
