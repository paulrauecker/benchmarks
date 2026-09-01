# Physics/math golden Q&A dataset

One JSON object per line in `physics_qa.jsonl`. A one-line working example is in the matching *.example.jsonl -- copy it as a starting point:

```json
{"id": "kinematics-01", "input": "A ball is dropped from 20m. How long until it hits the ground? (g=9.8 m/s^2)", "target": "2.02 s", "open_ended": false}
```

- `target`: a canonical answer. Scored by `normalized_match` (numeric-tolerant,
  looks for an `ANSWER:` line first) unless `open_ended: true`, in which case
  the pinned judge model (`judge:` in models.yaml) grades it. Prefer
  deterministic scoring wherever the answer has a canonical form — it's free
  and has no judge self-preference bias.
