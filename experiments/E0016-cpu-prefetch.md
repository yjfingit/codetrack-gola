# E0016: CPU prefetch correctness check

The evaluation pipeline already supports a CPU producer path through `num_workers` and
`num_io_threads`. The new `causal_clip_accept_prefetch` mixin sets 8 workers and 4 I/O threads;
it does not alter model state or prediction logic.

On `bike2left`, the prefetch run produced exactly the serial metrics:

- serial: SR 0.8134, PR 0.9455, NPR 0.9650;
- prefetch: SR 0.8134, PR 0.9455, NPR 0.9650.

The short single-sequence run took 22.99s including worker startup, so it is not a speed claim for
one short sequence. It is retained for full evaluation where multiple sequences keep the CPU
workers occupied. `io_wait` stayed around 0.06 in the logged batches.
