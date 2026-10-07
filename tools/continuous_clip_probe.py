"""Small learned-routing clip probe for CodeTrack temporal training.

The probe uses continuous LasHeR-train clips and paired synthetic faults, but never supplies the
fault mask as the SATR route. Only the first frame receives its initialization box; later Kalman
updates use the frozen GOLA head's previous prediction. Tracking loss decides whether a candidate
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
    ap.add_argument("--diagnosis-checkpoint", type=Path,
                    default=ROOT / "_probe/diagnosis_stage1_mixed.safetensors")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--diag-steps", type=int, default=120)
    ap.add_argument("--abstain-threshold", type=float, default=0.25)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--clip-length", type=int, default=4)
    ap.add_argument("--val-count", type=int, default=3,
                    help="number of held-out clips excluded from training")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from safetensors.torch import load_file, save_file
    from trackit.core.transforms.dataset_norm_stats import get_dataset_norm_stats_transform
    from trackit.core.utils.siamfc_cropping import (
        apply_siamfc_cropping, apply_siamfc_cropping_to_boxes,
        get_siamfc_cropping_params)
    from trackit.models import ModelImplSuggestions
    from trackit.models.methods.GOLA.builder import build_GOLA_model
    from tools.preflight_acceptance import load_stage_config
    from trackit.data.methods.siamese_tracker_train.transform.default.plugin.box_with_score_map_label_gen import (
        positive_sample_assignment)
    from codetrack.criteria import _tracking_loss

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dataset = Path("/home/yangjuanfeng/lab/dataset/LasHeR")
    names = (dataset / "trainingsetList.txt").read_text().splitlines()[:10]
    # Prefer the existing challenge-covering train-10 list when available.
    manifest = ROOT / "_probe/codetrack_flow_validation/LasHeR_curated10/trainingsetList.txt"
    if manifest.exists():
        names = manifest.read_text().splitlines()

    normalize = get_dataset_norm_stats_transform("mm", inplace=True)
    size_z, size_x = np.array((112, 112)), np.array((224, 224))

    def frame_crop(seq, frame, output_size, area, box_xyxy):
        imgs = []
        for modality in ("visible", "infrared"):
            files = sorted((seq / modality).glob("*.jpg"))
            with Image.open(files[frame]) as im:
                imgs.append(np.array(im.convert("RGB"), copy=True))
        image = torch.from_numpy(np.concatenate(imgs, axis=-1)).permute(2, 0, 1).float().cuda()
        params = get_siamfc_cropping_params(box_xyxy, area, output_size)
        crop = apply_siamfc_cropping(image, output_size, params, "bilinear", False)[0]
        normalize(crop.div(255.0))
        crop_xyxy = apply_siamfc_cropping_to_boxes(np.asarray(box_xyxy, dtype=np.float64)[None], params)[0]
        x1, y1, x2, y2 = crop_xyxy
        crop_box = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5,
                             max(1e-3, x2 - x1), max(1e-3, y2 - y1)], dtype=np.float64)
        return crop, crop_box

    clips = []
    for name in names:
        seq = dataset / "trainingset" / name
        boxes = np.loadtxt(seq / "init.txt", delimiter=",", ndmin=2)
        count = min(int(args.clip_length), len(boxes),
                    len(list((seq / "visible").glob("*.jpg"))))
        if count < args.clip_length:
            continue
        zbox = boxes[0].copy(); zbox[2:] += zbox[:2]
        z, _ = frame_crop(seq, 0, size_z, 2.0, zbox)
        frame_rows = []
        for t in range(count):
            box = boxes[t].copy(); box[2:] += box[:2]
            x, crop_box = frame_crop(seq, t, size_x, 4.0, box)
            # A small curriculum gives the clip clean -> damaged -> severely damaged -> clean
            # frames. This preserves a clean paired target without making every frame synthetic.
            cor = x.clone()
            cx, cy, bw, bh = crop_box
            damage_fraction = (0.0, 0.18, 0.32, 0.0)[t % 4]
            x0 = max(0, int(round(cx - damage_fraction * bw)))
            x1 = min(224, int(round(cx + damage_fraction * bw)))
            y0 = max(0, int(round(cy - damage_fraction * bh)))
            y1 = min(224, int(round(cy + damage_fraction * bh)))
            if damage_fraction > 0:
                # Damage TIR only; RGB remains as side information.
                cor[3:, y0:y1, x0:x1] = 0
            frame_rows.append((x, cor, crop_box, damage_fraction))
        clips.append((name, z, frame_rows))
    if len(clips) < 4:
        raise RuntimeError(f"only {len(clips)} usable clips")

    # Cache GOLA fused tokens. This model has no temporal state during capture; the clip state
    # is introduced only in the CodeTrack training loop below.
    cfg = load_stage_config(str(ROOT / "config/GOLA/codetrack_s1/config.yaml"))
    cfg["model"]["codetrack"]["corruption_enabled"] = False
    cfg["model"]["codetrack"]["motion_enabled"] = False
    cfg["model"]["codetrack"]["memory_enabled"] = False
    model = build_GOLA_model(cfg, ModelImplSuggestions()).cuda().eval()
    model.load_state_dict(load_file(str(ROOT / "weights/gola_b224.bin")), strict=False)
    ct = model.codetrack
    if args.diagnosis_checkpoint.exists():
        ct.diagnosis.load_state_dict(load_file(str(args.diagnosis_checkpoint)), strict=True)

    def capture(batch_z, batch_x):
        d = {"z": batch_z, "x": batch_x, "d": batch_z,
             "z_feat_mask": torch.ones(batch_z.shape[0], 8, 8, dtype=torch.long, device="cuda"),
             "d_feat_mask": torch.ones(batch_z.shape[0], 8, 8, dtype=torch.long, device="cuda")}
        got = []
        h = ct.register_forward_pre_hook(
            lambda _m, _i, kwargs: got.append(kwargs["F_L"].detach().clone()), with_kwargs=True)
        with torch.no_grad():
            model.reset_sequence(); model(**d)
        h.remove()
        return got[-1]

    cached = []
    for name, z, rows in clips:
        clean_seq, cor_seq, boxes, damage = [], [], [], []
        for x, cor, box, damage_fraction in rows:
            f_clean = capture(z.unsqueeze(0), x.unsqueeze(0))
            f_cor = capture(z.unsqueeze(0), cor.unsqueeze(0))
            clean_seq.append(f_clean); cor_seq.append(f_cor); boxes.append(box)
            damage.append(damage_fraction)
        cached.append((name, torch.cat(clean_seq), torch.cat(cor_seq),
                       torch.tensor(np.asarray(boxes), device="cuda", dtype=torch.float32),
                       torch.tensor(damage, device="cuda", dtype=torch.float32)))
    del model

    # Rebuild with causal motion. Temporal memory stays off in this minimal recipe so the
    # first temporal experiment isolates motion-conditioned correction.
    cfg = load_stage_config(str(ROOT / "config/GOLA/codetrack_s1/config.yaml"))
    cfg["model"]["codetrack"]["corruption_enabled"] = False
    cfg["model"]["codetrack"]["motion_enabled"] = True
    cfg["model"]["codetrack"]["memory_enabled"] = False
    cfg["model"]["codetrack"]["template_protection"] = False
    cfg["model"]["codetrack"]["topk_tokens"] = 4
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
        pos = positive_sample_assignment(pixels, np.array([16, 16]), np.array([224, 224]))
        gt = xyxy / 224.0
        return {
            "num_positive_samples": torch.tensor([len(pos)], device="cuda", dtype=torch.float32),
            "positive_sample_batch_dim_indices": torch.zeros(len(pos), device="cuda", dtype=torch.long),
            "positive_sample_map_dim_indices": torch.as_tensor(pos, device="cuda", dtype=torch.long),
            "boxes": gt[None],
        }

    causal_by_name = {}
    with torch.no_grad():
        for name, clean_f, cor_f, boxes, damage in cached:
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
    split = max(1, len(cached) - int(args.val_count))
    train_cached = cached[:split]
    val_cached = cached[split:]
    if not val_cached:
        raise RuntimeError("val-count leaves no held-out clip")

    ct.train()
    for _ in range(int(args.diag_steps)):
        losses = []
        for _name, clean_f, cor_f, boxes, damage in train_cached:
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
        score = head_out["score_map"].detach().float().sigmoid().flatten(1)
        best = score.argmax(dim=-1)
        confidence = score.gather(1, best[:, None]).squeeze(1)
        boxes = head_out["boxes"].detach().float().reshape(score.shape[0], -1, 4)
        xyxy = boxes.gather(1, best[:, None, None].expand(-1, 1, 4)).squeeze(1) * 224.0
        x1, y1, x2, y2 = xyxy.unbind(-1)
        xywh = torch.stack([(x1 + x2) * 0.5, (y1 + y2) * 0.5,
                            (x2 - x1).clamp(min=1.0), (y2 - y1).clamp(min=1.0)], dim=-1)
        return confidence, xywh

    def step_clip(name, clean_f, cor_f, boxes, damage, training=True):
        ct.reset_sequence()
        losses, feature_gains, tracking_gains, drifts = [], [], [], []
        for t in range(args.clip_length):
            clean_tok = ct._split(clean_f[t:t + 1])["X_TIR"].detach()
            cor_tok = ct._split(cor_f[t:t + 1])["X_TIR"].detach()
            mask = make_mask(boxes[t], damage[t])
            # Ground truth initializes the first frame only. Later motion observations are
            # decoded by the frozen GOLA head and consumed causally by the next frame.
            out = ct(F_L=cor_f[t:t + 1],
                     gt_box_xywh=boxes[t:t + 1] if t == 0 else None,
                     image_size=torch.tensor([[224., 224.]], device="cuda"),
                     corruption_mask=mask, update_state=True,
                     preserve_state=(t > 0), observe_motion=(t == 0), eval_observe=(t > 0))
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
        for _name, clean_f, cor_f, boxes, damage in items:
            ct.reset_sequence()
            gains, drifts = [], []
            for t in range(args.clip_length):
                clean_tok = ct._split(clean_f[t:t + 1])["X_TIR"].detach()
                cor_tok = ct._split(cor_f[t:t + 1])["X_TIR"].detach()
                mask = make_mask(boxes[t], damage[t])
                out = ct(F_L=cor_f[t:t + 1],
                         gt_box_xywh=boxes[t:t + 1] if t == 0 else None,
                         image_size=torch.tensor([[224., 224.]], device="cuda"),
                         corruption_mask=mask, update_state=True,
                         preserve_state=(t > 0), observe_motion=(t == 0),
                         eval_observe=(t > 0))
                head_out = ct.head(out["X_final"])
                score, pred_box = predicted_observation(head_out)
                ct.notify_tracking_score(score, box_xywh=pred_box)
                target = tracking_targets(boxes[t])
                repaired_loss, _, _ = _tracking_loss(head_out, target)
                base_loss, _, _ = _tracking_loss(ct.head(cor_tok), target)
                track_values.append(float((base_loss - repaired_loss).detach()))
                q_values.append(float(out["q"].mean()))
                active_values.append(float((out["q"] >= args.abstain_threshold).float().mean()))
                d_in = (1 - F.cosine_similarity(cor_tok, clean_tok, dim=-1)).clamp(0, 2)
                d_out = (1 - F.cosine_similarity(out["X_final"], clean_tok, dim=-1)).clamp(0, 2)
                gains.append(float((d_in.mean() - d_out.mean()).detach()))
                drifts.append(float((out["X_final"] - cor_tok).square().mean().detach()))
            values.append((sum(gains) / len(gains), sum(drifts) / len(drifts)))
        return {"gain": sum(x[0] for x in values) / len(values),
                "healthy_drift": sum(x[1] for x in values) / len(values),
                "q_mean": sum(q_values) / len(q_values),
                "active_fraction": sum(active_values) / len(active_values),
                "tracking_gain": sum(track_values) / len(track_values)}

    for step in range(args.steps):
        rows = [step_clip(name, clean_f, cor_f, boxes, damage)
                for name, clean_f, cor_f, boxes, damage in train_cached]
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
                                                   "motion.uncertainty_gain"))}}

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

    learned_eval = evaluate_learned(val_cached)
    train_eval = evaluate_learned(train_cached)
    report = {"probe": "continuous_clip_v1", "seed": args.seed,
              "clip_length": args.clip_length, "sequences": [x[0] for x in cached],
              "train_sequences": [x[0] for x in train_cached],
              "validation_sequences": [x[0] for x in val_cached],
              "best_step": best["step"], "best_score": best["score"],
              "learned_eval": learned_eval,
              "train_eval": train_eval,
              "causal_target_stats": causal_stats,
              "history": history, "checkpoint": str(args.save_checkpoint)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
