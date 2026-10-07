"""Causal replay training-paradigm probe for CodeTrack.

The probe uses continuous LasHeR-train clips and paired synthetic faults, but never supplies the
fault mask as the SATR route. Input crops follow frozen baseline predictions after clip
initialisation; annotations never enter the motion state. Tracking loss decides whether a candidate
correction is worth keeping, while clean paired features supervise repair and identity.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--save-checkpoint", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--diag-steps", type=int, default=120)
    ap.add_argument("--abstain-threshold", type=float, default=0.25)
    ap.add_argument("--residual-clip-ratio", type=float, default=0.05)
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--clip-length", type=int, default=4)
    ap.add_argument("--val-count", type=int, default=3,
                    help="number of held-out sequences excluded from training")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dataset", default="/home/yangjuanfeng/lab/dataset/LasHeR")
    ap.add_argument("--sequence-manifest", type=Path, default=ROOT / "experiments/train10.txt")
    ap.add_argument("--crop-policy", choices=("baseline", "gt_centered"), default="baseline")
    ap.add_argument("--clips-per-sequence", type=int, default=2)
    args = ap.parse_args()
    if args.clip_length < 4 or args.clips_per_sequence < 1:
        ap.error("use clip-length >= 4 and clips-per-sequence >= 1")

    import numpy as np
    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file, save_file
    from trackit.models import ModelImplSuggestions
    from trackit.models.methods.GOLA.builder import build_GOLA_model
    from tools.preflight_acceptance import load_stage_config
    from trackit.data.methods.siamese_tracker_train.transform.default.plugin.box_with_score_map_label_gen import (
        positive_sample_assignment)
    from codetrack.criteria import _tracking_loss

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    from tools.causal_clip_cache import build_replay_cache
    from trackit.runner.evaluation.distributed.tracker_evaluator.components.post_process.box_with_score_map import (
        PostProcessing_BoxWithScoreMap)
    cached = build_replay_cache(args)
    post = PostProcessing_BoxWithScoreMap(torch.device("cuda"), (16, 16), (224, 224), .45)
    post.start()

    # Rebuild with causal motion. Temporal memory stays off in this minimal recipe so the
    # first temporal experiment isolates motion-conditioned correction.
    cfg = load_stage_config(str(ROOT / "config/GOLA/codetrack_s1/config.yaml"))
    cfg["model"]["codetrack"]["corruption_enabled"] = False
    cfg["model"]["codetrack"]["motion_enabled"] = True
    cfg["model"]["codetrack"]["memory_enabled"] = False
    cfg["model"]["codetrack"]["template_protection"] = False
    cfg["model"]["codetrack"]["topk_tokens"] = int(args.topk)
    cfg["model"]["codetrack"]["residual_clip_ratio"] = float(args.residual_clip_ratio)
    cfg["model"]["codetrack"]["soft_route_training"] = True
    cfg["model"]["codetrack"]["abstain_enabled"] = True
    cfg["model"]["codetrack"]["abstain_threshold"] = float(args.abstain_threshold)
    cfg["model"]["codetrack"]["motion_route_scale"] = 0.0
    model = build_GOLA_model(cfg, ModelImplSuggestions()).cuda().float().eval()
    model.load_state_dict(load_file(str(ROOT / "weights/gola_b224.bin")), strict=False)
    ct = model.codetrack
    H = ct.H.matrix().detach()
    # Stage 1 uses the causal tracking-impact target.  The synthetic edit says where pixels were
    # changed, but not every changed token harms the head; fitting that injector mask was the
    # main source of the previous train/validation mismatch.
    diag_params = list(ct.diagnosis.parameters())
    for p in model.parameters():
        p.requires_grad_(False)
    for p in diag_params:
        p.requires_grad_(True)
    diag_opt = torch.optim.AdamW(diag_params, lr=5e-4, weight_decay=1e-5)

    def make_mask(box, damage_fraction):
        cx, cy, bw, bh = box
        if float(damage_fraction) <= 0.0:
            return torch.zeros(1, 256, device="cuda", dtype=torch.bool)
        x0, x1 = int(max(0, cx - damage_fraction * bw)) // 14, int(min(224, cx + damage_fraction * bw) + 13) // 14
        y0, y1 = int(max(0, cy - damage_fraction * bh)) // 14, int(min(224, cy + damage_fraction * bh) + 13) // 14
        m = torch.zeros(1, 256, device="cuda", dtype=torch.bool)
        for yy in range(max(0, y0), min(16, y1)):
            for xx in range(max(0, x0), min(16, x1)):
                m[:, yy * 16 + xx] = True
        return m

    def tracking_targets(box_xywh):
        cx, cy, bw, bh = box_xywh
        xyxy = torch.stack([cx - 0.5 * bw, cy - 0.5 * bh,
                            cx + 0.5 * bw, cy + 0.5 * bh])
        pixels = xyxy.detach().cpu().numpy()
        visible = pixels.clip(0., 224.)
        if bool((visible[2:] > visible[:2]).all()):
            pos = positive_sample_assignment(visible, np.array([16, 16]), np.array([224, 224]))
        else:
            pos = np.empty(0, dtype=np.int64)
        gt = xyxy / 224.0
        return {
            "num_positive_samples": torch.tensor([len(pos)], device="cuda", dtype=torch.float32),
            "positive_sample_batch_dim_indices": torch.zeros(len(pos), device="cuda", dtype=torch.long),
            "positive_sample_map_dim_indices": torch.as_tensor(pos, device="cuda", dtype=torch.long),
            "boxes": gt[None],
        }

    causal_by_name = {}
    with torch.no_grad():
        for name, clean_f, cor_f, boxes, damage, crop_params in cached:
            targets = []
            for t in range(args.clip_length):
                clean_tok = ct._split(clean_f[t:t + 1])["X_TIR"].detach()
                cor_tok = ct._split(cor_f[t:t + 1])["X_TIR"].detach()
                clean_head = ct.head(clean_tok.float())
                cor_head = ct.head(cor_tok.float())
                targets.append(model._causal_token_impact_target(
                    cor_tok, clean_tok, cor_head, clean_head,
                    tracking_targets=tracking_targets(boxes[t]), max_tokens=256,
                    min_ratio=0.25))
            causal_by_name[name] = torch.cat(targets, dim=0)
    causal_stats = {
        "mean": float(torch.cat(list(causal_by_name.values())).mean()),
        "positive_fraction": float((torch.cat(list(causal_by_name.values())) > 0).float().mean()),
        "strong_fraction": float((torch.cat(list(causal_by_name.values())) >= 0.5).float().mean()),
        "max": float(torch.cat(list(causal_by_name.values())).max()),
    }
    print(json.dumps({"causal_target_stats": causal_stats}), flush=True)
    sequence_order = list(dict.fromkeys(row[0].rsplit(":", 1)[0] for row in cached))
    if not 0 < args.val_count < len(sequence_order):
        raise ValueError("val-count must leave both train and held-out sequences")
    val_names = set(sequence_order[-args.val_count:])
    train_cached = [r for r in cached if r[0].rsplit(":", 1)[0] not in val_names]
    val_cached = [r for r in cached if r[0].rsplit(":", 1)[0] in val_names]
    assert not ({r[0].rsplit(":", 1)[0] for r in train_cached} & val_names)
    if not val_cached:
        raise RuntimeError("val-count leaves no held-out clip")

    ct.train()
    for _ in range(int(args.diag_steps)):
        losses = []
        for _name, clean_f, cor_f, boxes, damage, crop_params in train_cached:
            for t in range(args.clip_length):
                c = ct._split(cor_f[t:t + 1]); k = ct._split(clean_f[t:t + 1])
                tpl = torch.cat([c["Z_RGB"], c["Z_TIR"], c["Z_on"], c["D_TIR"]], dim=1)
                ctx = ct.template_pool(tpl.mean(dim=1)).detach()
                d = ct.diagnosis(c["X_TIR"].detach(), c["X_RGB"].detach(), H, template_context=ctx)
                causal = causal_by_name[_name][t:t + 1].to(d["q_logits"].device)
                if float(damage[t]) > 0.0:
                    l_diag = F.binary_cross_entropy_with_logits(
                        d["q_logits"].float(), causal.float(),
                        pos_weight=d["q_logits"].new_tensor(8.0))
                    positive = causal > 0.0
                    negative = ~positive
                    if bool(positive.any()) and bool(negative.any()):
                        l_rank = F.relu(0.25 - d["q_logits"][positive].mean()
                                         + d["q_logits"][negative].mean())
                    else:
                        l_rank = l_diag.new_zeros(())
                    losses.append(l_diag + 0.25 * l_rank)
                else:
                    # These frames are known clean by construction.  This is a safe negative
                    # example because it is a paired synthetic clean view, not an assumption
                    # about an arbitrary natural LasHeR frame.
                    losses.append(F.binary_cross_entropy_with_logits(
                        d["q_logits"].float(), torch.zeros_like(d["q_logits"])))
        dl = torch.stack(losses).mean()
        diag_opt.zero_grad(set_to_none=True); dl.backward(); torch.nn.utils.clip_grad_norm_(diag_params, 1.0); diag_opt.step()
    ct.train()
    # Diagnosis has already been fitted to the causal target above.  Keep it frozen while SATR
    # learns the repair direction; otherwise tracking loss rewards the detector for declaring
    # every token suspicious, which was the failure mode of the previous joint run.
    for p in ct.diagnosis.parameters():
        p.requires_grad_(False)
    train_params = list(ct.satr.parameters())
    if ct.motion is not None:
        train_params += list(ct.motion.prior.parameters()) + [ct.motion.uncertainty_gain]
    for p in train_params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=1e-5)
    history, best = [], {"score": -float("inf"), "step": -1, "state": None}

    def predicted_observation(head_out):
        decoded = post(head_out)
        xyxy = decoded["box"]
        xywh = torch.cat([(xyxy[:, :2] + xyxy[:, 2:]) * .5,
                          (xyxy[:, 2:] - xyxy[:, :2]).clamp_min(.001)], -1)
        return decoded["confidence"], xywh

    def rank_auc(scores, labels):
        labels = labels.to(torch.bool).reshape(-1)
        scores = scores.reshape(-1).float()
        if not bool(labels.any()) or not bool((~labels).any()):
            return None
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(labels.cpu().numpy(), scores.cpu().numpy()))

    def step_clip(name, clean_f, cor_f, boxes, damage, crop_params, training=True):
        ct.reset_sequence()
        losses, feature_gains, tracking_gains, drifts = [], [], [], []
        for t in range(args.clip_length):
            clean_tok = ct._split(clean_f[t:t + 1])["X_TIR"].detach()
            cor_tok = ct._split(cor_f[t:t + 1])["X_TIR"].detach()
            mask = make_mask(boxes[t], damage[t])
            # No current annotation enters the state. Rebase the posterior before
            # prediction, then consume the window-penalised accepted head box.
            out = ct(F_L=cor_f[t:t + 1],
                     gt_box_xywh=None,
                     search_crop_params=crop_params[t:t + 1],
                     image_size=torch.tensor([[224., 224.]], device="cuda"),
                     corruption_mask=mask, update_state=True,
                     preserve_state=(t > 0), observe_motion=False, eval_observe=True)
            repaired_head = ct.head(out["X_final"])
            base_head = ct.head(cor_tok)
            target = tracking_targets(boxes[t])
            track_repaired, _, _ = _tracking_loss(repaired_head, target)
            with torch.no_grad():
                track_base, _, _ = _tracking_loss(base_head, target)
            score, pred_box = predicted_observation(repaired_head)
            ct.notify_tracking_score(score, box_xywh=pred_box)

            d_in = (1 - F.cosine_similarity(cor_tok, clean_tok, dim=-1)).clamp(0, 2)
            d_out = (1 - F.cosine_similarity(out["X_final"], clean_tok, dim=-1)).clamp(0, 2)
            causal = causal_by_name[name][t:t + 1].to(out["q_logits"].device)
            # The causal target is continuous.  Values just above zero are weak numerical
            # gains, not reliable evidence that a token should be rewritten; using a strict
            # impact threshold keeps the healthy identity set non-empty.
            bad = causal >= 0.5
            healthy = ~bad
            drift = ((out["X_final"][healthy] - cor_tok[healthy]).square().mean()
                     if bool(healthy.any()) else out["X_final"].sum() * 0.0)
            if bool(bad.any()):
                feature_gain = d_in[bad].mean() - d_out[bad].mean()
                rec_loss = d_out[bad].mean()
                q_loss = F.binary_cross_entropy_with_logits(
                    out["q_logits"].float(), causal.float())
            else:
                feature_gain = d_in.new_zeros(())
                rec_loss = d_out.new_zeros(())
                q_loss = out["q_logits"].sum() * 0.0
            tracking_gain = track_base - track_repaired
            # Real tracking loss decides whether the learned route is useful. The margin term
            # explicitly pushes harmful corrections back toward the identity path.
            loss = (track_repaired + 0.5 * F.relu(track_repaired - track_base.detach())
                    + 0.2 * rec_loss + 1.0 * drift + 0.05 * q_loss)
            losses.append(loss)
            feature_gains.append(feature_gain.detach())
            tracking_gains.append(tracking_gain.detach())
            drifts.append(drift.detach())
        loss = torch.stack(losses).mean()
        return (loss, float(torch.stack(feature_gains).mean()),
                float(torch.stack(tracking_gains).mean()), float(torch.stack(drifts).mean()))

    @torch.no_grad()
    def evaluate_learned(items):
        ct.eval()
        values, q_values, active_values, track_values = [], [], [], []
        auc_scores, auc_labels = [], []
        per_clip, natural_tracks, synthetic_tracks, iou_values = [], [], [], []
        for _name, clean_f, cor_f, boxes, damage, crop_params in items:
            ct.reset_sequence()
            gains, drifts, local_tracks, local_ious = [], [], [], []
            for t in range(args.clip_length):
                clean_tok = ct._split(clean_f[t:t + 1])["X_TIR"].detach()
                cor_tok = ct._split(cor_f[t:t + 1])["X_TIR"].detach()
                mask = make_mask(boxes[t], damage[t])
                out = ct(F_L=cor_f[t:t + 1],
                         gt_box_xywh=None,
                         search_crop_params=crop_params[t:t + 1],
                         image_size=torch.tensor([[224., 224.]], device="cuda"),
                         corruption_mask=mask, update_state=True,
                         preserve_state=(t > 0), observe_motion=False,
                         eval_observe=True)
                head_out = ct.head(out["X_final"])
                score, pred_box = predicted_observation(head_out)
                ct.notify_tracking_score(score, box_xywh=pred_box)
                target = tracking_targets(boxes[t])
                repaired_loss, _, _ = _tracking_loss(head_out, target)
                base_head = ct.head(cor_tok)
                base_loss, _, _ = _tracking_loss(base_head, target)
                tg = float((base_loss - repaired_loss).detach())
                track_values.append(tg); local_tracks.append(tg)
                (synthetic_tracks if float(damage[t]) > 0 else natural_tracks).append(tg)
                def selected_iou(head):
                    corners = post(head)["box"][0]
                    truth = target["boxes"][0] * 224.
                    inter = (torch.minimum(corners[2:], truth[2:]) -
                             torch.maximum(corners[:2], truth[:2])).clamp_min(0.).prod()
                    union = ((corners[2:] - corners[:2]).clamp_min(0.).prod() +
                             (truth[2:] - truth[:2]).clamp_min(0.).prod() - inter)
                    return float(inter / union.clamp_min(1e-6))
                ig = selected_iou(head_out) - selected_iou(base_head)
                local_ious.append(ig); iou_values.append(ig)
                q_values.append(float(out["q"].mean()))
                active_values.append(float((out["q"] >= args.abstain_threshold).float().mean()))
                d_in = (1 - F.cosine_similarity(cor_tok, clean_tok, dim=-1)).clamp(0, 2)
                d_out = (1 - F.cosine_similarity(out["X_final"], clean_tok, dim=-1)).clamp(0, 2)
                causal = causal_by_name[_name][t:t + 1].to(out["q"].device)
                bad = causal >= 0.5
                healthy = ~bad
                if bool(bad.any()):
                    gains.append(float((d_in[bad].mean() - d_out[bad].mean()).detach()))
                else:
                    gains.append(0.0)
                drift = ((out["X_final"][healthy] - cor_tok[healthy]).square().mean()
                         if bool(healthy.any()) else out["X_final"].sum() * 0.0)
                drifts.append(float(drift.detach()))
                auc_scores.append(out["q"].detach().flatten())
                auc_labels.append((causal >= 0.5).flatten())
            values.append((sum(gains) / len(gains), sum(drifts) / len(drifts)))
            per_clip.append({"clip": _name, "tracking_gain": sum(local_tracks) / len(local_tracks),
                             "selected_box_iou_gain": sum(local_ious) / len(local_ious)})
        return {"gain": sum(x[0] for x in values) / len(values),
                "healthy_drift": sum(x[1] for x in values) / len(values),
                "q_mean": sum(q_values) / len(q_values),
                "active_fraction": sum(active_values) / len(active_values),
                "tracking_gain": sum(track_values) / len(track_values),
                "q_auc": rank_auc(torch.cat(auc_scores), torch.cat(auc_labels)),
                "natural_tracking_gain": sum(natural_tracks) / max(1, len(natural_tracks)),
                "synthetic_tracking_gain": sum(synthetic_tracks) / max(1, len(synthetic_tracks)),
                "selected_box_iou_gain": sum(iou_values) / max(1, len(iou_values)),
                "per_clip": per_clip}

    for step in range(args.steps):
        rows = [step_clip(*item) for item in train_cached]
        loss = torch.stack([r[0] for r in rows]).mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(train_params, 1.0); opt.step()
        if step % 25 == 0 or step == args.steps - 1:
            feature_gain = sum(r[1] for r in rows) / len(rows)
            tracking_gain = sum(r[2] for r in rows) / len(rows)
            drift = sum(r[3] for r in rows) / len(rows)
            row = {"step": step, "loss": float(loss.detach()),
                   "feature_gain": feature_gain, "tracking_gain": tracking_gain,
                   "healthy_drift": drift, "grad": float(grad)}
            history.append(row); print(json.dumps(row), flush=True)
            validation = evaluate_learned(val_cached)
            print(json.dumps({"step": step, "validation": validation}), flush=True)
            ct.train()
            score = validation["tracking_gain"] + 0.1 * validation["gain"] \
                    - 0.1 * validation["healthy_drift"]
            if score > best["score"]:
                best = {"score": score, "step": step,
                        "state": {k: v.detach().cpu().clone() for k, v in ct.state_dict().items()
                                  if k.startswith(("satr.", "diagnosis.", "motion.prior.",
                                                   "motion.uncertainty_gain", "template_pool.",
                                                   "H."))}}

    args.save_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    save_file(best["state"], str(args.save_checkpoint))
    partial = {k[len("satr."):]: v for k, v in best["state"].items() if k.startswith("satr.")}
    ct.satr.load_state_dict(partial, strict=True)
    partial = {k[len("diagnosis."):]: v for k, v in best["state"].items()
               if k.startswith("diagnosis.")}
    ct.diagnosis.load_state_dict(partial, strict=True)
    partial = {k[len("motion."):]: v for k, v in best["state"].items()
               if k.startswith("motion.")}
    if partial:
        ct.motion.load_state_dict(partial, strict=False)
    partial = {k[len("template_pool."):]: v for k, v in best["state"].items()
               if k.startswith("template_pool.")}
    if partial:
        ct.template_pool.load_state_dict(partial, strict=True)
    if "H.H" in best["state"]:
        ct.H.H.data.copy_(best["state"]["H.H"].to(ct.H.H))

    learned_eval = evaluate_learned(val_cached)
    import os
    previous_ablation = os.environ.get("CODETRACK_ABLATE_MOTION")
    os.environ["CODETRACK_ABLATE_MOTION"] = "1"
    try:
        motion_off_eval = evaluate_learned(val_cached)
    finally:
        if previous_ablation is None:
            os.environ.pop("CODETRACK_ABLATE_MOTION", None)
        else:
            os.environ["CODETRACK_ABLATE_MOTION"] = previous_ablation
    train_eval = evaluate_learned(train_cached)
    all_eval = evaluate_learned(cached)
    report = {"probe": "causal_replay_v1", "crop_policy": args.crop_policy,
              "configuration": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "seed": args.seed,
              "clip_length": args.clip_length, "sequences": [x[0] for x in cached],
              "train_sequences": [x[0] for x in train_cached],
              "validation_sequences": [x[0] for x in val_cached],
              "best_step": best["step"], "best_score": best["score"],
              "learned_eval": learned_eval, "motion_off_eval": motion_off_eval,
              "scope": "off-policy frozen-baseline crops; not candidate closed-loop PR/SR",
              "target_scope": "causal impact of paired synthetic faults; natural faults are not labelled",
              "train_eval": train_eval,
              "all_eval": all_eval,
              "causal_target_stats": causal_stats,
              "history": history, "checkpoint": str(args.save_checkpoint)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    post.stop()


if __name__ == "__main__":
    main()
