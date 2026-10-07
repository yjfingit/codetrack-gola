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
    ap.add_argument("--decoder-type", choices=("neural_bp", "syndrome_bp"), default="neural_bp")
    ap.add_argument("--tracking-quality", choices=("self", "fixed_reference", "fixed_gt"), default="self")
    ap.add_argument("--observation-recipe", choices=("legacy_blackout", "physical_mix"), default="legacy_blackout")
    ap.add_argument("--validation-degradation", choices=("seen", "heldout", "natural"), default="seen")
    ap.add_argument("--native-validation", action="store_true",
                    help="also evaluate untouched held-out LasHeR observations")
    ap.add_argument("--save-observation-examples", action="store_true")
    args = ap.parse_args()
    if args.clip_length < 4 or args.clips_per_sequence < 1:
        ap.error("use clip-length >= 4 and clips-per-sequence >= 1")
    if args.observation_recipe == "physical_mix" and args.crop_policy != "baseline":
        ap.error("physical observations require student-predicted crops")

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
    native_cached = []
    if args.native_validation:
        from copy import copy
        native_args = copy(args)
        native_args.sequence_manifest = args.output.parent / "native_sequences.txt"
        native_args.sequence_manifest.write_text("\n".join(
            Path(args.sequence_manifest).read_text().splitlines()[-args.val_count:]) + "\n")
        native_args.label_prefix = "native/"
        native_args.validation_degradation = "natural"
        native_args.output = args.output.with_name(args.output.stem + ".native_cache.json")
        native_cached = build_replay_cache(native_args)
    post = PostProcessing_BoxWithScoreMap(torch.device("cuda"), (16, 16), (224, 224), .45)
    post.start()

    # Rebuild with causal motion. Temporal memory stays off in this minimal recipe so the
    # first temporal experiment isolates motion-conditioned correction.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cfg = load_stage_config(str(ROOT / "config/GOLA/codetrack_s1/config.yaml"))
    cfg["model"]["codetrack"]["corruption_enabled"] = False
    cfg["model"]["codetrack"]["motion_enabled"] = True
    cfg["model"]["codetrack"]["memory_enabled"] = False
    cfg["model"]["codetrack"]["template_protection"] = False
    cfg["model"]["codetrack"]["topk_tokens"] = int(args.topk)
    cfg["model"]["codetrack"]["residual_clip_ratio"] = float(args.residual_clip_ratio)
    cfg["model"]["codetrack"]["soft_route_training"] = True
    if args.observation_recipe == "physical_mix":
        cfg["model"]["codetrack"].update(soft_route_training=False,
                                        match_inference_route_training=True)
    cfg["model"]["codetrack"]["abstain_enabled"] = True
    cfg["model"]["codetrack"]["abstain_threshold"] = float(args.abstain_threshold)
    cfg["model"]["codetrack"]["motion_route_scale"] = 0.0
    if args.decoder_type == "syndrome_bp":
        cfg["model"]["codetrack"].update(
            decoder_type="syndrome_bp", h_layout="binary_cycles", num_checks=256,
            h_links_per_check=4, h_min_col_degree=4, h_free_edge_frac=0.,
            detection_prior=.05, bp_iterations=3, bp_damping=0.)
    model = build_GOLA_model(cfg, ModelImplSuggestions()).cuda().float().eval()
    model.load_state_dict(load_file(str(ROOT / "weights/gola_b224.bin")), strict=False)
    ct = model.codetrack
    if args.observation_recipe == "physical_mix":
        def assert_student_conditions(_module, _inputs, kwargs):
            for key in ("gt_box_xywh", "corruption_mask", "image_corruption_mask",
                        "was_corrupted", "route_q_override", "clean_tokens", "x_clean"):
                if kwargs.get(key) is not None:
                    raise RuntimeError(f"supervision-only condition entered student: {key}")
        ct.register_forward_pre_hook(assert_student_conditions, with_kwargs=True)
    H = ct.H.matrix().detach()
    # Stage 1 uses the causal tracking-impact target.  The synthetic edit says where pixels were
    # changed, but not every changed token harms the head; fitting that injector mask was the
    # main source of the previous train/validation mismatch.
    diag_params = list(ct.diagnosis.parameters())
    if args.decoder_type == "syndrome_bp":
        diag_params += list(ct.motion.prior.parameters()) + [ct.motion.uncertainty_gain]
    for p in model.parameters():
        p.requires_grad_(False)
    for p in diag_params:
        p.requires_grad_(True)
    diag_opt = torch.optim.AdamW(diag_params, lr=5e-4, weight_decay=1e-5)

    def tracking_targets(box_xywh, reference_head=None):
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
        target = {
            "num_positive_samples": torch.tensor([len(pos)], device="cuda", dtype=torch.float32),
            "positive_sample_batch_dim_indices": torch.zeros(len(pos), device="cuda", dtype=torch.long),
            "positive_sample_map_dim_indices": torch.as_tensor(pos, device="cuda", dtype=torch.long),
            "boxes": gt[None],
        }
        if args.tracking_quality in ("fixed_reference", "fixed_gt"):
            if reference_head is None:
                raise ValueError("fixed quality requires a frozen reference head")
            from codetrack.criteria import bbox_overlaps
            quality = torch.zeros_like(reference_head["score_map"])
            positions = target["positive_sample_map_dim_indices"]
            if positions.numel():
                selected = reference_head["boxes"].reshape(1, -1, 4)[0, positions]
                quality.flatten(1)[0, positions] = (1. if args.tracking_quality == "fixed_gt" else
                    bbox_overlaps(gt.expand(len(pos), -1), selected, is_aligned=True).detach())
            target["score_quality_map"] = quality.detach()
        return target

    causal_by_name = {}
    trusted_by_name = {}
    with torch.no_grad():
        for name, clean_f, cor_f, boxes, damage, crop_params in list(cached) + list(native_cached):
            targets = []
            for t in range(args.clip_length):
                clean_tok = ct._split(clean_f[t:t + 1])["X_TIR"].detach()
                cor_tok = ct._split(cor_f[t:t + 1])["X_TIR"].detach()
                clean_head = ct.head(clean_tok.float())
                cor_head = ct.head(cor_tok.float())
                targets.append(model._causal_token_impact_target(
                    cor_tok, clean_tok, cor_head, clean_head,
                    tracking_targets=tracking_targets(boxes[t], clean_head), max_tokens=256,
                    min_ratio=0.25))
            causal_by_name[name] = torch.cat(targets, dim=0)
            if args.observation_recipe == "physical_mix":
                source_metadata = cached.metadata if name in cached.metadata else native_cached.metadata
                trusted_by_name[name] = torch.tensor(
                    [r["teacher_reliable"] for r in source_metadata[name]],
                    device="cuda", dtype=torch.bool)[:, None].expand(-1, 256)
            else:
                trusted_by_name[name] = torch.ones_like(causal_by_name[name], dtype=torch.bool)
    causal_stats = {
        "mean": float(torch.cat([causal_by_name[r[0]] for r in cached]).mean()),
        "positive_fraction": float((torch.cat([causal_by_name[r[0]] for r in cached]) > 0).float().mean()),
        "strong_fraction": float((torch.cat([causal_by_name[r[0]] for r in cached]) >= 0.5).float().mean()),
        "max": float(torch.cat([causal_by_name[r[0]] for r in cached]).max()),
        "unedited_positive_fraction": float(torch.cat([
            causal_by_name[name][damage == 0].reshape(-1)
            for name, _, _, _, damage, _ in cached]).gt(0).float().mean()),
        "trusted_frame_fraction": float(torch.cat([v[:, 0] for v in trusted_by_name.values()]).float().mean()),
    }
    no_op_targets = [causal_by_name[name][torch.tensor(
        [r["reference_equals_received"] for r in cached.metadata[name]], device="cuda")].flatten()
        for name, *_ in cached]
    no_op = torch.cat(no_op_targets)
    causal_stats["no_op_positive_fraction"] = float(no_op.gt(0).float().mean()) if no_op.numel() else None
    if causal_stats["no_op_positive_fraction"] not in (0., None):
        raise RuntimeError("identical clean/corrupted features produced nonzero impact labels")
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
            if args.decoder_type == "syndrome_bp":
                ct.reset_sequence()
            for t in range(args.clip_length):
                c = ct._split(cor_f[t:t + 1]); k = ct._split(clean_f[t:t + 1])
                tpl = torch.cat([c["Z_RGB"], c["Z_TIR"], c["Z_on"], c["D_TIR"]], dim=1)
                ctx = ct.template_pool(tpl.mean(dim=1)).detach()
                if args.decoder_type == "syndrome_bp":
                    d = ct(F_L=cor_f[t:t + 1], image_size=cor_f.new_tensor([[224., 224.]]),
                           search_crop_params=crop_params[t:t + 1], observe_motion=False,
                           preserve_state=(t > 0), eval_observe=True)
                    decoded = post(ct.head(c["X_TIR"]))
                    corners = decoded["box"]
                    obs = torch.cat([(corners[:, :2] + corners[:, 2:]) * .5,
                                     (corners[:, 2:] - corners[:, :2]).clamp_min(.001)], -1)
                    ct.notify_tracking_score(decoded["confidence"], box_xywh=obs)
                else:
                    d = ct.diagnosis(c["X_TIR"].detach(), c["X_RGB"].detach(), H, template_context=ctx)
                causal = causal_by_name[_name][t:t + 1].to(d["q_logits"].device)
                if args.decoder_type == "syndrome_bp":
                    bits = (causal >= .5).float()
                    known = ((causal == 0.) | (causal >= .5)) & trusted_by_name[_name][t:t + 1]
                    parity_bits = torch.einsum("mn,bn->bm", (H > 0).float(), bits).remainder(2.)
                    check_known = torch.einsum("mn,bn->bm", (H > 0).float(), (~known).float()) == 0.
                    unary = F.binary_cross_entropy_with_logits(d["channel_logits"], bits, reduction="none")
                    posterior = F.binary_cross_entropy_with_logits(d["q_logits"], bits, reduction="none")
                    parity_target = (torch.where(check_known, parity_bits, torch.full_like(parity_bits, .5))
                                     if args.observation_recipe == "physical_mix" else parity_bits)
                    parity = F.binary_cross_entropy_with_logits(d["parity_logits"], parity_target, reduction="none")
                    loss = (((unary + posterior)[known].mean() if bool(known.any()) else unary.sum() * 0.) +
                            (parity.mean() if args.observation_recipe == "physical_mix" else
                             parity[check_known].mean() if bool(check_known.any()) else parity.sum() * 0.))
                    losses.append(loss)
                    continue
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
        if args.decoder_type == "syndrome_bp" and _ % 25 == 0:
            print(json.dumps({"diagnosis_step": _, "detection_loss": float(dl.detach())}), flush=True)
    ct.train()
    # Diagnosis has already been fitted to the causal target above.  Keep it frozen while SATR
    # learns the repair direction; otherwise tracking loss rewards the detector for declaring
    # every token suspicious, which was the failure mode of the previous joint run.
    for p in ct.diagnosis.parameters():
        p.requires_grad_(False)
    train_params = list(ct.satr.parameters())
    if ct.motion is not None and args.decoder_type != "syndrome_bp":
        train_params += list(ct.motion.prior.parameters()) + [ct.motion.uncertainty_gain]
    elif ct.motion is not None:
        for p in ct.motion.prior.parameters():
            p.requires_grad_(False)
        ct.motion.uncertainty_gain.requires_grad_(False)
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
        order = torch.argsort(scores)
        _, counts = torch.unique_consecutive(scores[order], return_counts=True)
        ends = counts.cumsum(0).to(scores.dtype)
        average_ranks = ends - (counts.to(scores.dtype) - 1.) * .5
        ranks = torch.empty_like(scores)
        ranks[order] = torch.repeat_interleave(average_ranks, counts)
        pos = labels.sum().to(scores.dtype)
        neg = (~labels).sum().to(scores.dtype)
        return float((ranks[labels].sum() - pos * (pos + 1.) / 2.) / (pos * neg))

    def step_clip(name, clean_f, cor_f, boxes, damage, crop_params, training=True):
        ct.reset_sequence()
        losses, feature_gains, tracking_gains, drifts = [], [], [], []
        for t in range(args.clip_length):
            clean_tok = ct._split(clean_f[t:t + 1])["X_TIR"].detach()
            cor_tok = ct._split(cor_f[t:t + 1])["X_TIR"].detach()
            # No current annotation enters the state. Rebase the posterior before
            # prediction, then consume the window-penalised accepted head box.
            out = ct(F_L=cor_f[t:t + 1],
                     gt_box_xywh=None,
                     search_crop_params=crop_params[t:t + 1],
                     image_size=torch.tensor([[224., 224.]], device="cuda"),
                     update_state=True,
                     preserve_state=(t > 0), observe_motion=False, eval_observe=True)
            repaired_head = ct.head(out["X_final"])
            base_head = ct.head(cor_tok)
            with torch.no_grad():
                target = tracking_targets(boxes[t], ct.head(clean_tok))
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
            trusted = trusted_by_name[name][t:t + 1]
            bad = (causal >= 0.5) & trusted
            healthy = (causal == 0.) & trusted
            preservation_set = healthy | (~trusted) if args.observation_recipe == "physical_mix" else healthy
            drift = ((out["X_final"][healthy] - cor_tok[healthy]).square().mean()
                     if bool(healthy.any()) else out["X_final"].sum() * 0.0)
            preservation = ((out["X_final"][preservation_set] - cor_tok[preservation_set]).square().mean()
                            if bool(preservation_set.any()) else out["X_final"].sum() * 0.)
            if bool(bad.any()):
                feature_gain = d_in[bad].mean() - d_out[bad].mean()
                rec_loss = d_out[bad].mean()
                q_loss = (F.binary_cross_entropy_with_logits(out["q_logits"].float(), causal.float())
                          if args.observation_recipe != "physical_mix" else out["q_logits"].sum() * 0.)
            else:
                feature_gain = d_in.new_zeros(())
                rec_loss = d_out.new_zeros(())
                q_loss = out["q_logits"].sum() * 0.0
            tracking_gain = track_base - track_repaired
            # Real tracking loss decides whether the learned route is useful. The margin term
            # explicitly pushes harmful corrections back toward the identity path.
            loss = (track_repaired + 0.5 * F.relu(track_repaired - track_base.detach())
                    + 0.2 * rec_loss + 1.0 * preservation + 0.05 * q_loss)
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
        channel_scores, parity_scores, parity_labels, oracle_scores, shuffled_scores = [], [], [], [], []
        per_clip, natural_tracks, synthetic_tracks, iou_values = [], [], [], []
        direction_values, write_values, labelled_count, total_count = [], [], 0, 0
        for _name, clean_f, cor_f, boxes, damage, crop_params in items:
            ct.reset_sequence()
            gains, drifts, local_tracks, local_ious = [], [], [], []
            for t in range(args.clip_length):
                clean_tok = ct._split(clean_f[t:t + 1])["X_TIR"].detach()
                cor_tok = ct._split(cor_f[t:t + 1])["X_TIR"].detach()
                out = ct(F_L=cor_f[t:t + 1],
                         gt_box_xywh=None,
                         search_crop_params=crop_params[t:t + 1],
                         image_size=torch.tensor([[224., 224.]], device="cuda"),
                         update_state=True,
                         preserve_state=(t > 0), observe_motion=False,
                         eval_observe=True)
                head_out = ct.head(out["X_final"])
                score, pred_box = predicted_observation(head_out)
                ct.notify_tracking_score(score, box_xywh=pred_box)
                target = tracking_targets(boxes[t], ct.head(clean_tok))
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
                trusted = trusted_by_name[_name][t:t + 1]
                known = ((causal == 0.) | (causal >= .5)) & trusted
                bad = (causal >= 0.5) & trusted
                healthy = (causal == 0.) & trusted
                residual = out["X_final"] - cor_tok
                written = residual.norm(dim=-1) > 1e-7
                write_values.append(float(written.float().mean()))
                if bool((bad & written).any()):
                    direction_values.append(float(F.cosine_similarity(
                        residual[bad & written], (clean_tok - cor_tok)[bad & written], dim=-1).mean()))
                labelled_count += int(known.sum())
                total_count += known.numel()
                if bool(bad.any()):
                    gains.append(float((d_in[bad].mean() - d_out[bad].mean()).detach()))
                else:
                    gains.append(0.0)
                drift = ((out["X_final"][healthy] - cor_tok[healthy]).square().mean()
                         if bool(healthy.any()) else out["X_final"].sum() * 0.0)
                drifts.append(float(drift.detach()))
                auc_scores.append(out["q"].detach()[known].flatten())
                auc_labels.append((causal >= 0.5)[known].flatten())
                if out.get("channel_logits") is not None:
                    from codetrack.syndrome_bp import decode_error_syndrome
                    support = H > 0
                    bits = (causal >= .5).float()
                    parity_bits = torch.einsum("mn,bn->bm", support.float(), bits).remainder(2.)
                    check_known = torch.einsum("mn,bn->bm", support.float(), (~known).float()) == 0.
                    channel_scores.append(out["channel_logits"].sigmoid()[known].flatten())
                    parity_scores.append(out["parity_logits"].sigmoid()[check_known].flatten())
                    parity_labels.append(parity_bits.bool()[check_known].flatten())
                    oracle_likelihood = torch.where(check_known, (2. * parity_bits - 1.) * 12., torch.zeros_like(parity_bits))
                    oracle_scores.append(decode_error_syndrome(
                        out["channel_logits"], support, oracle_likelihood)["q"][known].flatten())
                    shuffled_scores.append(decode_error_syndrome(
                        out["channel_logits"], support, out["parity_logits"].roll(17, -1))["q"][known].flatten())
            values.append((sum(gains) / len(gains), sum(drifts) / len(drifts)))
            per_clip.append({"clip": _name, "tracking_gain": sum(local_tracks) / len(local_tracks),
                             "selected_box_iou_gain": sum(local_ious) / len(local_ious)})
        mechanism = {}
        if channel_scores and torch.cat(channel_scores).numel():
            truth = torch.cat(auc_labels).float()
            mechanism = {
                "channel_auc": rank_auc(torch.cat(channel_scores), truth),
                "oracle_syndrome_auc": rank_auc(torch.cat(oracle_scores), truth),
                "shuffled_syndrome_auc": rank_auc(torch.cat(shuffled_scores), truth),
                "parity_auc": rank_auc(torch.cat(parity_scores), torch.cat(parity_labels)),
                "q_brier": float((torch.cat(auc_scores) - truth).square().mean()),
                "channel_brier": float((torch.cat(channel_scores) - truth).square().mean()),
                "parity_brier": float((torch.cat(parity_scores) - torch.cat(parity_labels).float()).square().mean()) if torch.cat(parity_labels).numel() else None,
                "constant_parity_brier": float((torch.cat(parity_labels).float() - torch.cat(parity_labels).float().mean()).square().mean()) if torch.cat(parity_labels).numel() else None}
        return {"gain": sum(x[0] for x in values) / len(values),
                "healthy_drift": sum(x[1] for x in values) / len(values),
                "q_mean": sum(q_values) / len(q_values),
                "active_fraction": sum(active_values) / len(active_values),
                "tracking_gain": sum(track_values) / len(track_values),
                "q_auc": rank_auc(torch.cat(auc_scores), torch.cat(auc_labels)),
                "natural_tracking_gain": sum(natural_tracks) / max(1, len(natural_tracks)),
                "synthetic_tracking_gain": sum(synthetic_tracks) / max(1, len(synthetic_tracks)),
                "selected_box_iou_gain": sum(iou_values) / max(1, len(iou_values)),
                "per_clip": per_clip, "syndrome_mechanism": mechanism,
                "repair_direction_cosine": sum(direction_values) / len(direction_values) if direction_values else None,
                "actual_write_fraction": sum(write_values) / max(1, len(write_values)),
                "q_labelled_fraction": labelled_count / max(1, total_count)}

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
    native_eval = evaluate_learned(native_cached) if native_cached else None
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
              "native_eval": native_eval,
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
