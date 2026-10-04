"""Which module talks to which, through exactly what tensor.

Builds the producer -> consumer matrix from the *recorded* boundary tensors of a real training
step (no manual bookkeeping), then classifies each edge:

  FWD   forward-only information (shape/values)
  GRAD  a path the training objective actually pushes gradient through
  DET   the tensor is detached -> forward only, no gradient (intentional or not)
"""
from __future__ import annotations
import os, sys, re
from collections import defaultdict
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPORT = "/tmp/dataflow3.txt"

def parse():
    ins, outs, grads = [], [], {}
    for line in open(REPORT):
        m = re.match(r"\s+\[(IN |OUT)\] (\S+)\s+(.*)$", line.rstrip("\n"))
        if m:
            kind, name, rest = m.group(1).strip(), m.group(2), m.group(3)
            (ins if kind == "IN" else outs).append((name, rest))
            continue
        g = re.match(r"\s+\[\s*(?:IN|OUT)\s*\]\s+(\S+)\s+\|grad\|=\s*([\d.eE+-]+)\s+rel=\s*([\d.eE+-]+)(.*)$",
                     line.rstrip("\n"))
        if g:
            grads[g.group(1)] = (float(g.group(2)), g.group(4).strip())
    return ins, outs, grads

# producer of a given consumer input, derived from the interface names seen in the report
EDGES = [
    # (producer module / source, consumer, tensor, note)
    ("GOLA trunk (corrupted)", "CodeTrack._split", "F_L (B,768,768)", "cor_fused"),
    ("CodeTrack._split", "SyndromeDiagnosis", "X_t (B,256,768)", "TIR-position search tokens"),
    ("CodeTrack._split", "SyndromeDiagnosis", "X_aux (B,256,768)", "RGB-position search tokens"),
    ("CodeTrack.template_pool", "SyndromeDiagnosis", "template_context (B,768)", "Z_RGB|Z_TIR|Z_on|D_TIR pooled"),
    ("ParityCheckMatrix", "SyndromeDiagnosis", "H_bar (64,256)", "row-normalised, learnable edge weights"),
    ("KalmanMotionPrior", "TemporalMemory", "uncertainty (B,)", "u_t"),
    ("KalmanMotionPrior", "TemporalMemory", "mahalanobis (B,)", "innovation / sqrt(S)"),
    ("KalmanMotionPrior", "H_RoutedSparseRefiner", "motion_map (B,1,16,16)", "log-space attention BIAS"),
    ("KalmanMotionPrior", "NoiseModulatedDenoiser", "motion (B,2)", "[mean(map), u_t] frame cond"),
    ("KalmanMotionPrior", "TemplateProtectionGate", "uncertainty (B,)", "gate input"),
    ("SyndromeDiagnosis", "TemporalMemory", "reliability = 1-q (B,256)", "detached token reliability"),
    ("SyndromeDiagnosis", "H_RoutedSparseRefiner", "q (B,256)", "TopK selector AND residual gate"),
    ("SyndromeDiagnosis", "NoiseModulatedDenoiser", "syndrome (B,64)", "frame condition"),
    ("SyndromeDiagnosis", "TemplateProtectionGate", "mean q (B,)", "corruption summary"),
    ("TemporalMemory", "H_RoutedSparseRefiner", "prior_tokens (B,8,128)", "TGS output -> memory_proj"),
    ("TemporalMemory", "NoiseModulatedDenoiser", "memory (B,128)", "mean over prior tokens"),
    ("TemporalMemory", "CodeTrack", "trc_confidence (B,3)", "supervised by L_trc"),
    ("TemporalMemory", "CodeTrack", "frame_reliability (B,1)", "supervised by L_mem"),
    ("ParityCheckMatrix", "H_RoutedSparseRefiner", "neighbour_index (256,8)", "H^T H shared-check geometry"),
    ("CodeTrack.template_pool", "H_RoutedSparseRefiner", "template_pool (B,768)", "per-suspect condition"),
    ("CodeTrack._split", "H_RoutedSparseRefiner", "X_aux (B,256,768)", "cross-modal evidence"),
    ("H_RoutedSparseRefiner", "NoiseModulatedDenoiser", "X_rec (B,256,768)", "refined tokens -> denoiser input"),
    ("H_RoutedSparseRefiner", "CodeTrack.condition_proj", "context (B,32,256)", "scattered to suspect rows"),
    ("CodeTrack._split", "CodeTrack.condition_proj", "X_aux (B,256,768)", "condition = [ctx | aux]"),
    ("NoiseModulatedDenoiser", "MeanVarCompletion", "X_denoised (B,256,768)", "X_final"),
    ("NoiseModulatedDenoiser", "GOLA head", "X_final (B,256,768)", "MAIN PATH to tracking"),
    ("NoiseModulatedDenoiser", "TemplateProtectionGate", "recovery_confidence (B,)", "1-cos(X_final,X_t)"),
    ("MeanVarCompletion", "criterion L_align", "pred_mean/pred_logvar", "vs clean token stats"),
    ("TemplateProtectionGate", "inference updater", "c_t (B,)", "score>0.84 AND c_t>tau_c"),
    ("GOLA head (clean branch)", "criterion L_track^clean", "score_map/boxes", "teacher head keeps a graph"),
    ("GOLA trunk (clean branch)", "criterion e_feat/e_task", "X_clean (B,256,768) DETACHED", "diagnosis target"),
    ("GOLA head (corrupted)", "criterion L_track^corr", "score_map/boxes", "MAIN supervision"),
    ("ParityCheckMatrix", "criterion L_diag", "H_bar", "support defines the check target"),
]

def main():
    import json
    d = json.load(open("/tmp/dataflow_grads.json"))
    G = {}
    for r in d["grads"]:
        G.setdefault(r["name"], r["grad_norm"])

    print("=" * 112)
    print("MODULE INTERACTION MATRIX  (recorded from a real training step, batch=2)")
    print("=" * 112)
    print(f"  {'producer':32s} -> {'consumer':24s} {'tensor':26s} note")
    print("  " + "-" * 108)
    for prod, cons, ten, note in EDGES:
        print(f"  {prod:32s} -> {cons:24s} {ten:26s} {note}")

    print("\n" + "=" * 112)
    print("DOES GRADIENT ACTUALLY FLOW ALONG EACH EDGE?")
    print("=" * 112)
    print(f"  {'interface tensor':38s} {'|grad|':>11s}  {'verdict':16s} role")
    print("  " + "-" * 108)
    rows = [
        ("KalmanMotionPrior.motion_map",       "prior map -> refiner attention bias"),
        ("KalmanMotionPrior.uncertainty",      "u_t -> TRC gate / denoiser / template gate"),
        ("KalmanMotionPrior.mahalanobis",      "innovation / sqrt(S) -> TRC gate"),
        ("SyndromeDiagnosis.H_bar",            "H_bar: learnable edge weights"),
        ("SyndromeDiagnosis.C_obs",            "check-node observation"),
        ("SyndromeDiagnosis.C_ref",            "check-node reference"),
        ("SyndromeDiagnosis.s",                "syndrome -> denoiser frame condition"),
        ("SyndromeDiagnosis.q",                "q -> TopK selector + residual gate + gate summary"),
        ("TemporalMemory.trc_confidence",      "TRC per-frame reliability"),
        ("TemporalMemory.prior_tokens",        "TGS prior tokens -> refiner / denoiser"),
        ("H_RoutedSparseRefiner.motion_map",   "motion_map AS SEEN by the refiner"),
        ("H_RoutedSparseRefiner.delta",        "recovery residual dX"),
        ("H_RoutedSparseRefiner.X_rec",        "refined tokens -> denoiser input"),
        ("H_RoutedSparseRefiner.context",      "routed context -> denoiser condition"),
        ("H_RoutedSparseRefiner.suspect_score", "TopK severity weights"),
        ("NoiseModulatedDenoiser.condition",   "condition AS SEEN by the denoiser"),
        ("NoiseModulatedDenoiser.X_denoised",  "X_final -> head  (MAIN PATH)"),
        ("MeanVarCompletion.pred_mean",        "mean-var completion -> L_align"),
        ("TemplateProtectionGate.c_t",         "template gate decision"),
    ]
    dead = []
    for key, role in rows:
        if key not in G:
            print(f"  {key:38s} {'--':>11s}  {'not recorded':16s} {role}")
            continue
        gn = G[key]
        if gn == 0:
            v = "DEAD"; dead.append(key)
        elif gn < 1e-6:
            v = "weak (<1e-6)"
        else:
            v = "training"
        print(f"  {key:38s} {gn:11.3e}  {v:16s} {role}")

    print("\n  Detached by design (forward-only, must NOT backprop):")
    for t, why in [("X_clean / clean_tokens", "diagnosis target: teacher must not receive student gradients"),
                   ("memory bank (B,3,128)", "a rolling buffer, not a parameterised path"),
                   ("reliability = 1-q", "stops L_diag from fighting the recovery objective"),
                   ("c_t / should_update", "inference selection statistic, not a loss term")]:
        print(f"    {t:26s} {why}")

    print("\n  Loss breakdown:")
    for k, v in sorted(d["losses"].items()):
        print(f"    {k:24s} {v:12.6f}")
    print("\n  DEAD interfaces:", dead if dead else "none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
