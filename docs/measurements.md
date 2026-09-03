# Measured throughput and timings

Empirical reference data from real runs, kept so future runs can be *budgeted*
rather than guessed at, and so a surprising score can be checked against what
the hardware was actually doing at the time.

Every number here was measured, not estimated. Where a figure is contaminated
by a known bug it is labelled as such rather than quietly dropped — the
contaminated numbers are the reason several of the fixes exist.

---

## Hardware / endpoints

| Box | Model | Params | Quant | Slots |
|---|---|---|---|---|
| ServerS (`servers.cougar-diatonic.ts.net:8080`) | Qwen3.8-Flash-Next | 176.9B MoE | UD-Q4_K_XL | 4 |
| proteus (`proteus:8080`) | Qwen3.8-27B | 27B dense | UD-Q4_K_XL | 4 |

ServerS launch flags (2026-09-03):

```
-ngl 99 -sm layer -fa on -ctk q8_0 -ctv q8_0 -ub 256 -b 2048
-np 4 -c 786432 --kv-unified --cont-batching --cache-reuse 256
--rope-scaling yarn --rope-freq-scale 3.0 --yarn-orig-ctx 262144 --jinja
```

Note `--kv-unified`: context is a shared 786432-token pool allocated on demand,
**not** statically divided `ctx_size / parallel`. `/props` reports
`n_ctx = 262144`, which is the YaRN *original/trained* context, not the
extrapolated pool — do not read it as a per-slot budget on this build.

---

## Decode throughput

The single most useful number for budgeting. Derived from clean (low-retry)
tasks as `total output tokens / wall seconds`:

| Box | Aggregate (4 slots) | Per slot |
|---|---|---|
| ServerS (177B MoE) | **~29 tok/s** | ~8.3 tok/s |
| proteus (27B dense) | **~48 tok/s** | ~12 tok/s |

ServerS's per-slot figure is independently corroborated by `llama-server`'s own
`slot print_timing` output (`tg = 7.95–8.20 t/s`), so it is trustworthy.

Rough planning rule: **wall seconds ≈ total output tokens / aggregate tok/s.**

---

## ServerS — Qwen3.8-Flash-Next

### Timing probe, 2026-09-03 (`logs/20260903-143250-timing-probe`)

`reasoning_effort: medium`, `temperature: 0` (greedy), `max_tokens: 8192`,
`max_connections: 4`, `--limit 10` per task.

| Task | Generations | Wall | s/gen | out_tok mean | median | max | Aggregate tok/s |
|---|---|---|---|---|---|---|---|
| ifeval | 10 | 5:23 | 32.3 | 1068 | 754 | 2428 | 33.1 |
| gpqa_diamond | 40 (10×4ep) | 1:13:44 | 110.6 | 1918 | 1334 | **8192** | 17.3 |
| mmlu_pro | 10 | 6:03 | 36.3 | 1045 | 902 | 2825 | 28.8 |
| math | 10 | 6:08 | 36.8 | 1053 | 926 | 2245 | 28.6 |
| aime2025 | 26/40 (killed) | ~2:58:36 | **412** ⚠ | 2490 | 1619 | 4838 | — |

Scores (tiny n — indicative only): ifeval 0.844, gpqa_diamond 0.975,
mmlu_pro 0.800, math 0.900 / 0.900 / 0.800.

⚠ **`aime2025`'s 412 s/gen is retry-contaminated, not a real model speed.** At
2490 mean output tokens and 29 tok/s it should cost **~86 s/gen** — a 4.8x
inflation caused by inspect's 600s default request timeout cancelling
long generations and restarting them from scratch (56 retries for 26
completions). Fixed by `timeout: 1800` in `models.yaml`. Use ~86–150 s/gen for
planning, not 412.

`gpqa_diamond` is genuinely ~3x a normal sample: its `max = 8192` shows it hits
the `max_tokens` cap, and cap-hitting generations at 8.3 tok/s take ~1024s —
which also exceeded the old 600s timeout.

### Full `core_math`, 2026-09-01/02 (`logs/20260901-160012-qwen38`)

⚠ **Superseded — do not cite these scores.** Ran with `max_tokens: 1024` and
**no** `reasoning_effort` (i.e. unconstrained reasoning). The tight token cap
truncated reasoning on most samples; `gpqa_diamond` at 0.164 is a truncation
artifact, not a capability measurement. Retained for timing only.

| Task | Samples | Duration | s/sample |
|---|---|---|---|
| ifeval | 541 | 5h45m | 38.3 |
| gpqa_diamond | 792 (198×4ep) | 6h09m | 28.0 |
| mmlu_pro | 1000 | 5h17m | 19.0 |
| math | 500 | 8h21m | 60.1 |
| aime2025 | 120 (30×4ep) | 4h00m | 120.0 |

~26.5h active (34h16m wall, including dead time from the 5000-sample `math`
bug). Per-sample times are *faster* than the 2026-09-03 probe precisely because
`max_tokens: 1024` cut every generation short.

Scores (invalid, for the record): ifeval 0.643, gpqa_diamond 0.164,
mmlu_pro 0.608, math 0.724 / 0.688 / 0.672, aime2025 0.042.

---

## proteus — Qwen3.8-27B

### Smoke, 2026-09-02 (`logs/20260902-215950-smoke`)

`reasoning_effort: none`, `temperature: 0`, `max_tokens: 4096`,
`max_connections: 4`, `--limit 20`.

| Task | Generations | Wall | s/gen | Tokens (I / O) | Aggregate tok/s |
|---|---|---|---|---|---|
| ifeval | 20 | 1:55 | 5.8 | 1,014 / 5,485 | 47.7 |
| gpqa_diamond | 80 (20×4ep) | 1:01:49 | 46.4 | 20,270 / 179,826 | 48.5 |
| humaneval | 20 | 1:23 | 4.2 | 2,855 / 3,228 | 38.9 |

Scores: ifeval 0.958, gpqa_diamond 0.688, humaneval 1.000.

### Cost of leaving `reasoning_effort` unset

Same box, same task, effort **unset** (unconstrained reasoning):
`gpqa_diamond` ran at **~482 s/gen** (25/80 in 3h20m) versus **46.4 s/gen** at
`reasoning_effort: none` — an **~8x** penalty. Unset does not mean "some sane
default"; it means unbounded. Always pin it.

---

## OpenRouter reference

`minimax-m3-free`, full `core_math`, $0.00 (free tier):
ifeval 0.808±0.017, gpqa_diamond 0.798±0.022, mmlu_pro 0.821±0.012,
math 0.859±0.005 / 0.810±0.006, aime2025 0.158±0.052.

The third `math` submetric (`expression_exact_match_sympy`) is invalid for this
run — it predates the `antlr4-python3-runtime` fix and reported 0.000 for every
sample. The other two math metrics are string-normalisation based and unaffected.

---

## Gotchas these runs exposed

1. **`gpqa_diamond` defaults to `epochs = 4`** inside `inspect_evals`
   (`DEFAULT_EPOCHS = 4`). Omitting `epochs` silently quadruples the task.
   The other tasks here default to 1.
2. **`mbpp` bakes in `epochs: 5`** (with pass@k reducers), so it silently
   costs 5x its stated limit -- same class of trap as `gpqa_diamond`'s
   `epochs = 4`.
3. **`scicode`'s `limit` counts MAIN PROBLEMS, not generations.** Each
   problem loops `generate` once per subproblem — 65 problems / 291
   subproblems, mean 4.48, median 3, **max 15** — so `limit: 8` is ~36
   generations. Subproblems are sequential within a problem and the
   conversation accumulates, so prompts grow as it goes, and cost is highly
   variable (a 15-subproblem draw costs ~5x a median one). It also needs
   `gdown` (test data is on Google Drive) and pulls a ~1GB `test_data.h5`
   on first run.
4. **`math` has no built-in cap** — its dataset is the full 5,000-sample MATH
   test split, not MATH-500. Uncapped it can run 24h+ on a slow backend.
5. **`mmlu_pro`, `math` and `ds1000` shuffle their datasets UNSEEDED**
   (`hf_dataset(shuffle=True)` with `seed` defaulting to `None`) — verified
   live: two consecutive loads return different items. Without
   `args: {shuffle: false}` plus a pinned `sample_shuffle`, every model is
   scored on a *different* random subsample. `gpqa_diamond` and `ifeval` don't
   shuffle samples at all.
6. **inspect's default request timeout is 600s.** At ~8 tok/s a full 8192-token
   generation takes ~1024s, so long generations get cancelled and retried from
   scratch, compounding into a retry storm rather than just running long once.
7. **Continuous batching makes runs non-deterministic even at temperature 0** —
   batch composition changes floating-point reduction order. Expect small
   run-to-run variation on self-hosted models, and treat it as a noise floor
   below which differences cannot be resolved.

---

## Saturated benchmarks (do not use for measurement)

`humaneval` returned **1.000** on the 27B at n=20 and again on Flash at n=2;
`mbpp` returned **1.000** at n=2x5. A benchmark pinned at ceiling cannot
resolve a difference at any sample size. Both are kept only in `smoke`, where
the job is proving the Docker sandbox works. `ifeval` at 0.84-0.96 is close
behind. Discriminative signal lives in `gpqa_diamond`, `math`, `mmlu_pro`, and
(newly added) `ds1000` / `scicode`.

## Statistical power (what these sample sizes can actually resolve)

Detectable difference at 95% confidence / 80% power, worst case p≈0.5:

| n per model | Unpaired | Paired (same items) |
|---|---|---|
| 20 | ~44pp | ~31pp |
| 30 | ~36pp | ~26pp |
| 50 | ~28pp | ~20pp |
| 130 (pooled) | — | ~12pp |
| 1,000 | ~6pp | ~3pp |

Unpaired: `n = 3.92 / d²`. Paired (McNemar): `n ≈ 7.85 · δ / d²`, where δ is the
between-model disagreement rate.

Implications:
- Small suites resolve **architecture-scale** gaps (20pp+), not quant-scale ones.
- Distinguishing quantisations (typically 1–3pp) needs ~1,000–2,000 *paired*
  samples — 10–20h per quant here. Use `llama-perplexity`'s KL-divergence
  against a higher-precision baseline instead; it compares full output
  distributions rather than a thresholded right/wrong bit and needs orders of
  magnitude less compute.
- **Ceiling effects kill signal**: `humaneval` at 1.000 and `ifeval` at
  0.84–0.96 have no headroom. Discriminative signal lives in `gpqa_diamond`,
  `math`, and `mmlu_pro`.
