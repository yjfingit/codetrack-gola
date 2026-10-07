# E0009: Legacy Continuous-Clip SATR Probes

- Date: 2026-10-06
- Git commit / dirty state: current worktree contains subsequent edits; probe source is `tools/continuous_clip_probe.py`.
- Status: complete; diagnostic only
- Hypothesis: a four-frame sequence can train SATR with temporal state and improve corrupted-token feature recovery.
- Structure and config: GOLA features cached for clean and locally erased TIR crops; a second CodeTrack model enables Kalman and temporal memory, with four Tanner neighbors selected per token in some variants.
- Training recipe and sampling: four consecutive frames from ten LasHeR-train sequences. The script trains diagnosis separately, then trains SATR over clips. Crucially, the SATR optimization path sets q from the synthetic corruption box mask (`route_q_override`) and supplies ground-truth box observations to motion. These are oracle conditions, not deployment-equivalent training.
- Data and split: ten LasHeR-train clips; several saved probes reuse these clips and vary abstention/configuration.
- Seed: 42.
- GPU: GPU used by the recorded probe is not reliably captured in the JSON; do not infer it.
- Checkpoint: variants include `_probe/continuous_clip75_v*.safetensors`, `_probe/continuous_clip_joint*.safetensors`, and `_probe/continuous_clip_ranked.safetensors`.
- Training log: per-step summaries are embedded in the corresponding JSON files; console log is not retained.
- Evaluation log: per-probe JSON files listed above.
- Reproduction command: see `tools/continuous_clip_probe.py`; exact options and input checkpoint must be read from the matching saved JSON/command history before claiming exact replay.
- Results: oracle-routed training showed positive corrupted-feature gain, but learned-q evaluation remained weak or negative on several variants. Examples: `continuous_clip_abstain25.json` learned gain +0.00548 with healthy drift 0.13378; `continuous_clip_joint.json` +0.01048 with drift 0.28006; `continuous_clip75_v4.json` -0.00341 with drift 0.30020.
- Mechanism evidence: the gap between oracle-route training and learned-route evaluation shows the probe did not establish that q learns when to correct. Ground-truth motion observations further prevent this from proving causal motion use.
- Conclusion: do not use these probes as a training-paradigm win or as evidence for temporal correction in the paper.
- Failure analysis and next action: the next clip experiment must use predicted/causal motion observations and learned q in the actual training path; compare it to random-pair training at matched updates, sequences, seed, and model parameters. Hold out validation sequences from LasHeR-train for checkpoint/threshold selection.
