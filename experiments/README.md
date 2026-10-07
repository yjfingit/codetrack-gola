# CodeTrack Experiments

This directory is the durable experiment record. Every new experiment gets a sequential ID,
one Markdown record, and a row in `summary.md`. Keep raw logs and checkpoints at their existing
paths; link them here rather than moving large artifacts.

Required record fields: date, git commit and dirty state, hypothesis, structure/config changes,
training recipe and sampling, data split, GPU, checkpoint, log, metrics, mechanism evidence,
decision, and next action. Mark unavailable evidence explicitly. Do not describe a short probe as
a converged or full training run.

Use `template.md` for new records.
