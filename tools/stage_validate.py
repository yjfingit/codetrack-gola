"""Per-stage acceptance gates for the staged CodeTrack recipe.

A stage is not "done" because the training loop finished; it is done when the *mechanism* the
stage exists to establish has been demonstrated.  This module implements that distinction:

  HARD gates  -- failing one must stop the pipeline (the next stage's starting point is invalid)
  SOFT gates  -- reported as WARN; a weak-but-not-broken result should not block progress, but it
                 must never be silently rounded up to "pass"

Design rules
------------
* Metrics come from `tools/stage_metrics.py`, which parses the real training log.  Nothing here
  re-derives a metric with a different formula than the one the criterion optimises.
* Recovery improvement is judged on ``d_final < d_input`` (``Error/gain_total > 0``), never on
  "the parameters moved".  A branch can train vigorously and still make the features worse --
  measured on this codebase, `d_after / d_input` reached 4.8 at 280 optimizer updates.
* Diagnosis quality is judged on ``AUROC/ AUPRC(q, corruption_mask)``.  The soft-target variant
  is threshold-dependent and is reported, never decisive.
* A frozen group must move by exactly 0.  "Nearly zero" is not a pass; it means the freeze is
  leaking through a path nobody intended.

Usage
-----
    from tools.stage_validate import validate_stage
    report = validate_stage(1, log_path=..., deltas=..., extra=...)
    if report.hard_failed: ...   # stop
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from tools.stage_metrics import StageLog

# --------------------------------------------------------------------------- thresholds
# These are engineering gates, not paper claims.  They are deliberately stated once, here, so a
# reviewer can see exactly what was asserted -- and so that "making the test green" would require
# editing this visible block rather than quietly relaxing an inline comparison.
DIAG_AUROC_SOFT = 0.60          # AUROC(q, corruption_mask) sanity floor
DIAG_Q_STD_MIN = 1e-3           # q must not collapse to a constant
GAIN_TOTAL_MIN = 5e-3           # d_input - d_final must exceed fp32 noise (~1e-7)
FREEZE_DELTA_MAX = 0.0          # a frozen group moves by exactly zero
MOVE_DELTA_MIN = 1e-6           # a trainable group must actually move
CLEAN_SR_DROP_MAX = 0.3         # absolute percentage points, stage 2 gate
LR_TRAVERSED_MIN = 0.5          # final lr must be below (first lr - lr_min) * this


@dataclass
class Check:
    name: str
    status: str                 # PASS / WARN / FAIL
    detail: str = ""

    @property
    def failed(self) -> bool:
        return self.status == "FAIL"


@dataclass
class StageReport:
    stage: int
    name: str
    checks: List[Check] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    deltas: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str = "") -> None:
        self.checks.append(Check(name, status, detail))

    @property
    def hard_failed(self) -> bool:
        return any(c.failed for c in self.checks)

    @property
    def warned(self) -> bool:
        return any(c.status == "WARN" for c in self.checks)

    @property
    def decision(self) -> str:
        if self.hard_failed:
            return "NO-GO"
        return "GO (with WARN)" if self.warned else "GO"

    def to_dict(self) -> dict:
        return {"stage": self.stage, "name": self.name, "decision": self.decision,
                "checks": [{"name": c.name, "status": c.status, "detail": c.detail}
                           for c in self.checks],
                "metrics": self.metrics, "deltas": self.deltas, "notes": self.notes}

    def to_text(self) -> str:
        bar = "=" * 64
        lines = [bar, f"CodeTrack Stage {self.stage} Report -- {self.name}", bar]
        for c in self.checks:
            lines.append(f"  [{c.status:4s}] {c.name}" + (f"   {c.detail}" if c.detail else ""))
        if self.metrics:
            lines.append("")
            lines.append("Metrics:")
            for k in sorted(self.metrics):
                lines.append(f"  {k:<28} = {self.metrics[k]}")
        if self.deltas:
            lines.append("")
            lines.append("Parameter delta (max|change|):")
            for k in sorted(self.deltas):
                lines.append(f"  {k:<28} = {self.deltas[k]:.3e}")
        if self.notes:
            lines.append("")
            lines.extend(f"  note: {n}" for n in self.notes)
        lines.append("")
        lines.append(f"FINAL: {self.decision}")
        lines.append(bar)
        return "\n".join(lines)


# --------------------------------------------------------------------------- shared checks
def _common_checks(rep: StageReport, log: StageLog, deltas: Optional[Dict[str, float]],
                   must_move: Dict[str, str], must_freeze: Dict[str, str]) -> None:
    # Numerical health.  ``grad_norm`` is intentionally allowed to be nan on the micro-steps that
    # do not end an accumulation window -- that is the harness reporting "no optimizer step yet",
    # not a failure -- so only ``loss`` is checked for finiteness here.
    loss_series = log.series("loss")
    bad_loss = [v for v in loss_series if v != v or v in (float("inf"), float("-inf"))]
    rep.add("no NaN/Inf in loss", "FAIL" if bad_loss else "PASS",
            f"{len(bad_loss)} bad of {len(loss_series)}" if bad_loss else f"{len(loss_series)} steps")
    gnorm = [v for v in log.series("grad_norm") if v == v and abs(v) != float("inf")]
    rep.add("grad_norm finite at accumulation boundaries", "PASS" if gnorm else "WARN",
            f"max={max(gnorm):.3f}" if gnorm else "never logged finite")

    # Optimizer time axis: the cosine must actually be traversed, otherwise the stage ran on a
    # horizon that does not match its own update budget.
    lr = log.learning_rates()
    if lr:
        rep.metrics.update(lr)
        span = lr["lr_first"] - 1e-6
        drop = lr["lr_first"] - lr["lr_last"]
        rep.add("cosine horizon traversed", "PASS" if drop > LR_TRAVERSED_MIN * span else "FAIL",
                f"{lr['lr_first']:.3e} -> {lr['lr_last']:.3e}")

    if deltas is None:
        rep.add("parameter deltas available", "WARN", "no before/after snapshot supplied")
        return

    rep.deltas.update(deltas)
    for label, key in must_move.items():
        d = deltas.get(key)
        if d is None:
            rep.add(f"{label} moved", "WARN", "group not present in the snapshot")
            continue
        rep.add(f"{label} moved", "PASS" if d > MOVE_DELTA_MIN else "FAIL", f"max|delta|={d:.3e}")
    for label, key in must_freeze.items():
        d = deltas.get(key)
        if d is None:
            rep.add(f"{label} frozen", "WARN", "group not present in the snapshot")
            continue
        rep.add(f"{label} frozen (exactly zero change)",
                "PASS" if d <= FREEZE_DELTA_MAX else "FAIL", f"max|delta|={d:.3e}")


def _recovery_checks(rep: StageReport, log: StageLog, hard: bool) -> None:
    """d_input / d_before / d_final, all as ``1 - cos`` on corrupted tokens only."""
    for key in ("Error/d_input", "Error/d_before", "Error/d_after", "Error/gain_total",
                "Error/gain_refiner"):
        v = log.at_fraction(key, 1.0, ema=False)
        if v is not None:
            rep.metrics[key] = v
    gain = log.at_fraction("Error/gain_total", 1.0, ema=False)
    d_in = log.at_fraction("Error/d_input", 1.0, ema=False)
    d_after = log.at_fraction("Error/d_after", 1.0, ema=False)
    if gain is None:
        rep.add("recovery gain measurable", "FAIL",
                "Error/gain_total absent: check that input_tokens/corruption_mask are exposed")
        return
    status = "PASS" if gain > GAIN_TOTAL_MIN else ("FAIL" if hard else "WARN")
    rep.add("recovery improves on its input (d_final < d_input)", status,
            f"d_input={d_in} d_final={d_after} gain_total={gain:+.4f} (need > {GAIN_TOTAL_MIN:g})")


def _diagnosis_checks(rep: StageReport, log: StageLog) -> None:
    auroc = log.at_fraction("Error/q_auroc_mask", 1.0, ema=False)
    auprc = log.at_fraction("Error/q_auprc_mask", 1.0, ema=False)
    q_std = log.at_fraction("Error/q_std", 1.0, ema=False)
    q_pos = log.at_fraction("Error/q_pos_mean", 1.0, ema=False)
    q_neg = log.at_fraction("Error/q_neg_mean", 1.0, ema=False)
    for k, v in (("AUROC(q, mask)", auroc), ("AUPRC(q, mask)", auprc), ("q_std", q_std),
                 ("q_pos_mean", q_pos), ("q_neg_mean", q_neg)):
        if v is not None:
            rep.metrics[k] = v
    if auroc is None:
        rep.add("diagnosis AUROC measurable", "FAIL", "Error/q_auroc_mask absent")
    else:
        rep.add("diagnosis AUROC(q, corruption_mask) above chance", "PASS" if auroc > 0.5 else "FAIL",
                f"AUROC={auroc:.4f}")
        rep.add(f"diagnosis AUROC sanity floor ({DIAG_AUROC_SOFT})",
                "PASS" if auroc > DIAG_AUROC_SOFT else "WARN",
                f"AUROC={auroc:.4f} -- sanity check only, not a paper claim")
    if q_std is not None:
        rep.add("q has not collapsed to a constant", "PASS" if q_std > DIAG_Q_STD_MIN else "FAIL",
                f"q_std={q_std:.3e}")
    if q_pos is not None and q_neg is not None:
        rep.add("q separates damaged from untouched tokens", "PASS" if q_pos > q_neg else "FAIL",
                f"pos={q_pos:.4f} neg={q_neg:.4f}")


def _gate_check(rep: StageReport, gate_init: Optional[float], gate_now: Optional[float]) -> None:
    if gate_init is None or gate_now is None:
        rep.add("residual gate probed", "WARN", "gate value not supplied")
        return
    moved = abs(gate_now - gate_init)
    import math
    rep.metrics["gate_init"] = gate_init
    rep.metrics["gate_final"] = gate_now
    rep.metrics["gate_sigmoid_init"] = 1.0 / (1.0 + math.exp(-gate_init))
    rep.metrics["gate_sigmoid_final"] = 1.0 / (1.0 + math.exp(-gate_now))
    rep.add("residual gate is not stuck", "PASS" if moved > 1e-3 else "WARN",
            f"{gate_init:.4f} -> {gate_now:.4f} (sigmoid {rep.metrics['gate_sigmoid_init']:.5f}"
            f" -> {rep.metrics['gate_sigmoid_final']:.5f})")


# --------------------------------------------------------------------------- per stage
def validate_stage(stage: int, log_path: str,
                   deltas: Optional[Dict[str, float]] = None,
                   gate_init: Optional[float] = None,
                   gate_now: Optional[float] = None,
                   tracking: Optional[Dict[str, float]] = None,
                   baseline: Optional[Dict[str, float]] = None,
                   name: str = "") -> StageReport:
    """Run the gates for one stage and return a report.

    ``tracking`` / ``baseline`` carry PR/NPR/SR measured on a fixed diagnostic subset; they are
    supplied by the orchestrator after it runs the evaluation pipeline, and are optional so that
    a stage can still be gated on mechanism alone when evaluation is unavailable.
    """
    log = StageLog(log_path).parse()
    rep = StageReport(stage=stage, name=name or f"stage{stage}")
    rep.metrics["updates_logged"] = float(log.updates_logged)
    rep.metrics["micro_steps_logged"] = float(len(log.rows))

    if stage == 1:
        _common_checks(rep, log, deltas,
                       must_move={"H / diagnosis / refiner / denoiser": "codetrack."},
                       must_freeze={"DINOv2 backbone": "blocks.",
                                    "GOLA head": "head.",
                                    "LoRA adapters": "lora",
                                    "token_type_embed": "token_type_embed"})
        _diagnosis_checks(rep, log)
        _recovery_checks(rep, log, hard=True)
        _gate_check(rep, gate_init, gate_now)

    elif stage == 2:
        _common_checks(rep, log, deltas,
                       must_move={"CodeTrack": "codetrack.",
                                  "LoRA adapters": "lora",
                                  "GOLA head": "head."},
                       must_freeze={"DINOv2 backbone": "blocks."})
        _diagnosis_checks(rep, log)
        _recovery_checks(rep, log, hard=False)
        if tracking and baseline:
            clean_now = tracking.get("clean_SR")
            clean_base = baseline.get("clean_SR")
            if clean_now is not None and clean_base is not None:
                drop = (clean_base - clean_now) * 100.0 if clean_base <= 1.0 else clean_base - clean_now
                rep.add("clean tracking does not degrade", "PASS" if drop < CLEAN_SR_DROP_MAX else "FAIL",
                        f"clean SR {clean_base} -> {clean_now} (drop {drop:.2f})")
            rob_now, rob_base = tracking.get("corrupted_SR"), baseline.get("corrupted_SR")
            if rob_now is not None and rob_base is not None:
                rep.add("robust tracking improved", "PASS" if rob_now >= rob_base else "WARN",
                        f"corrupted SR {rob_base} -> {rob_now} (WARN only: see docs for baseline)")
        else:
            rep.add("tracking metrics available", "WARN",
                    "no PR/NPR/SR supplied; stage gated on mechanism only")

    elif stage == 3:
        _common_checks(rep, log, deltas,
                       must_move={"CodeTrack": "codetrack.", "DINOv2 last-2 blocks": "blocks.1"},
                       must_freeze={"DINOv2 blocks 0-9": "blocks.0"})
        if tracking and baseline:
            sr_now, sr_base = tracking.get("SR"), baseline.get("SR")
            clean_now, clean_base = tracking.get("clean_SR"), baseline.get("clean_SR")
            gain = None
            if sr_now is not None and sr_base is not None:
                gain = sr_now - sr_base
                rep.add("S3 improves SR", "PASS" if gain > 0 else "FAIL",
                        f"SR {sr_base} -> {sr_now} (delta {gain:+.4f})")
            if clean_now is not None and clean_base is not None:
                drop = (clean_base - clean_now) * 100.0 if clean_base <= 1.0 else clean_base - clean_now
                rep.add("S3 does not degrade clean tracking",
                        "PASS" if drop < CLEAN_SR_DROP_MAX else "FAIL",
                        f"clean SR {clean_base} -> {clean_now} (drop {drop:.2f})")
            rep.notes.append("S3 is rejected (and S2 restored) if either check fails")
        else:
            rep.add("tracking metrics available", "FAIL",
                    "S3 cannot be accepted or rejected without PR/NPR/SR")

    elif stage == 4:
        _common_checks(rep, log, deltas,
                       must_move={"motion prior": "codetrack.motion.",
                                  "temporal memory": "codetrack.memory.",
                                  "spatial CodeTrack": "codetrack.refiner."},
                       must_freeze={"DINOv2 backbone": "blocks."})
        # Motion must be supervised and must be a function of history, not of the current GT.
        ml_first, ml_last = log.first_last("Loss/motion", ema=True)
        if ml_first is None:
            rep.add("motion loss computed", "FAIL", "Loss/motion absent (lambda_motion or target missing)")
        else:
            rep.metrics["Loss/motion first"] = ml_first
            rep.metrics["Loss/motion last"] = ml_last
            rep.add("motion loss computed", "PASS", f"{ml_first:.4f} -> {ml_last:.4f}")
            rep.add("motion loss trending down", "PASS" if ml_last < ml_first else "WARN",
                    f"delta {ml_last - ml_first:+.4f}")
        # Reliability must be able to express low trust; the D4 defect pinned it to [0.5, 0.73].
        for key in ("Error/memory_rel_min", "Error/memory_rel_max", "Error/memory_rel_mean",
                    "Error/memory_accepted_ratio"):
            v = log.at_fraction(key, 1.0, ema=False)
            if v is not None:
                rep.metrics[key] = v
        rel_min = log.at_fraction("Error/memory_rel_min", 1.0, ema=False)
        rel_max = log.at_fraction("Error/memory_rel_max", 1.0, ema=False)
        if rel_min is None or rel_max is None:
            rep.add("memory reliability range reported", "FAIL",
                    "Error/memory_rel_{min,max} absent; the D4 defect cannot be detected without them")
        else:
            rep.add("memory reliability spans a useful range",
                    "PASS" if (rel_max - rel_min) > 0.3 else "FAIL",
                    f"[{rel_min:.3f}, {rel_max:.3f}] span={rel_max - rel_min:.3f} "
                    f"(the D4 defect gave [0.500, 0.731])")
        if tracking and baseline:
            sr_now, sr_base = tracking.get("SR"), baseline.get("SR")
            if sr_now is not None and sr_base is not None:
                rep.add("temporal improves over spatial", "PASS" if sr_now > sr_base else "WARN",
                        f"SR {sr_base} -> {sr_now}")
            for attr in ("FM", "PO", "TO", "LI", "TC"):
                a_now, a_base = tracking.get(attr), baseline.get(attr)
                if a_now is not None and a_base is not None:
                    rep.add(f"attribute {attr} improved", "PASS" if a_now >= a_base else "WARN",
                            f"{a_base} -> {a_now}")
        else:
            rep.add("tracking metrics available", "FAIL",
                    "S4 cannot be accepted or rejected without PR/NPR/SR")
        rep.notes.append("S4 is not allowed to become the final model if it shows no gain")
    else:
        rep.add("known stage", "FAIL", f"stage {stage} has no gate definition")

    return rep


def write_report(rep: StageReport, reports_dir: str) -> None:
    os.makedirs(reports_dir, exist_ok=True)
    with open(os.path.join(reports_dir, f"stage{rep.stage}_report.json"), "w") as fh:
        json.dump(rep.to_dict(), fh, indent=2, sort_keys=True)
    with open(os.path.join(reports_dir, f"stage{rep.stage}_report.txt"), "w") as fh:
        fh.write(rep.to_text() + "\n")
