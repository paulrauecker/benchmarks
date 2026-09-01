# Structured extraction dataset

One JSON object per line in `extraction.jsonl`. Build this from real documents
you have already processed, where you know the correct output because you
produced it — 20-30 real cases beats a large synthetic set.

```json
{"id": "inv-001", "input": "<document text>", "schema": {"vendor": "string", "total": "number"}, "target": {"vendor": "Acme Corp", "total": 1234.56}}
```

- `input`: the document/prompt text given to the model.
- `schema` (optional): appended to the prompt so the model knows the field names.
- `target`: the ground-truth JSON. Nested objects/lists are supported; scoring
  flattens to dotted paths and lists are compared order-insensitively.

Scored by `field_f1` (precision/recall/F1 over fields) and `json_valid`
(bare parse rate) — see `src/llm_bench/tasks/scorers.py`.
