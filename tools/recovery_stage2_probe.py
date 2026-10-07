"""Stage-2 fixed-feature recovery gate using the stage-1 diagnosis checkpoint.

This probe trains only SATR.  It reports oracle routing and learned-q routing separately,
which prevents a good refiner from hiding a bad detector (or vice versa).
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
    ap.add_argument("--diagnosis-checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--save-checkpoint", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--topk", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-route", choices=("oracle", "learned"), default="oracle",
                    help="route used to train SATR; learned avoids oracle-to-inference mismatch")
    ap.add_argument("--causal-mask-ratio", type=float, default=0.25,
                    help="relative causal head-gain threshold used as the repair mask")
    ap.add_argument("--q-threshold", type=float, default=0.0,
                    help="set q below this value to zero before SATR; 0 disables abstention")
    ap.add_argument("--identity-weight", type=float, default=0.1,
                    help="weight for healthy-token identity loss")
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file, save_file
    from trackit.models import ModelImplSuggestions
    from trackit.models.methods.GOLA.builder import build_GOLA_model
    from tools.preflight_acceptance import load_stage_config
    from tools.verify_codetrack_initialization import real_batch

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    view = ROOT / "_probe/codetrack_flow_validation/LasHeR_curated10"
    names = (view / "trainingsetList.txt").read_text().splitlines()
    data = real_batch(view, names)
    cfg = load_stage_config(str(ROOT / "config/GOLA/codetrack_s1/config.yaml"))
    cfg["model"]["codetrack"]["corruption_enabled"] = False
    cfg["model"]["codetrack"]["topk_tokens"] = int(args.topk)
    cfg["model"]["codetrack"]["freeze_vote_bias"] = True
    cfg["model"]["codetrack"]["center_bp_logits"] = True
    cfg["model"]["codetrack"]["h_layout"] = "grid"
    cfg["model"]["codetrack"]["h_free_edge_frac"] = 0.0
    model = build_GOLA_model(cfg, ModelImplSuggestions()).cuda().float().eval()
    model.load_state_dict(load_file(str(ROOT / "weights/gola_b224.bin")), strict=False)
    ct = model.codetrack
    ct.diagnosis.load_state_dict(load_file(str(args.diagnosis_checkpoint)), strict=True)

    regions = {
        "center": (84, 140, 84, 140),
        "top_left": (28, 84, 28, 84),
        "top_right": (28, 84, 140, 196),
        "bottom_right": (140, 196, 140, 196),
    }

    def capture(input_data):
        got = []
        h = ct.register_forward_pre_hook(
            lambda _m, _i, kwargs: got.append(kwargs["F_L"].detach().clone()),
            with_kwargs=True)
        with torch.no_grad():
            model.reset_sequence()
            model(**input_data)
        h.remove()
        return got[-1]

    clean = capture(data)
    fused = {}
    for name, (y0, y1, x0, x1) in regions.items():
        damaged = dict(data)
        damaged["x"] = data["x"].clone()
        damaged["x"][:, 3:, y0:y1, x0:x1] = 0
        fused[name] = capture(damaged)

    def split(f):
        toks = ct._split(f)
        x_t, x_aux = toks["X_TIR"], toks["X_RGB"]
        template = torch.cat([toks["Z_RGB"], toks["Z_TIR"], toks["Z_on"], toks["D_TIR"]], dim=1)
        return x_t.detach(), x_aux.detach(), ct.template_pool(template.mean(dim=1)).detach()

    clean_t = split(clean)[0]
    H = ct.H.matrix().detach()
    grid = ct.cfg.grid
    parts = []
    for name, f in fused.items():
        y0, y1, x0, x1 = regions[name]
        mask = torch.zeros(clean_t.shape[0], grid * grid, device=clean_t.device, dtype=torch.bool)
        for yy in range(y0 // 14, min((y1 + 13) // 14, grid)):
            for xx in range(x0 // 14, min((x1 + 13) // 14, grid)):
                mask[:, yy * grid + xx] = True
        x_t, x_aux, template = split(f)
        # Sequence holdout: every spatial route is seen during fitting, while two complete
        # target sequences remain untouched for validation.
        n_train = max(1, int(round(x_t.shape[0] * 0.8)))
        # The injector mask identifies where pixels were edited.  The recovery target must
        # instead identify final head tokens whose clean replacement improves the head.  This
        # prevents the SATR loss from supervising a token that q correctly chose to ignore.
        with torch.no_grad():
            clean_head = model.head(clean_t.float())
            bad_head = model.head(x_t.float())
            causal_target = model._causal_token_impact_target(
                x_t, clean_t, bad_head, clean_head, min_ratio=float(args.causal_mask_ratio))
            causal_mask = causal_target > 0.0
        parts.append((name, x_t[:n_train], x_aux[:n_train], template[:n_train],
                      causal_mask[:n_train], clean_t[:n_train]))
        parts.append((name, x_t[n_train:], x_aux[n_train:], template[n_train:],
                      causal_mask[n_train:], clean_t[n_train:]))
    train = parts[0::2]
    valid = parts[1::2]

    for p in model.parameters():
        p.requires_grad_(False)
    for p in ct.satr.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(ct.satr.parameters(), lr=float(args.lr), weight_decay=1e-5)
    ct.satr.train()
    history = []
    best = {"score": -float("inf"), "step": -1, "state": None}

    def run_one(item, route_mode):
        _name, x_t, x_aux, template, mask, clean_ref = item
        with torch.no_grad():
            diag = ct.diagnosis(x_t, x_aux, H, template_context=template)
        q = mask.float() if route_mode == "oracle" else diag["q"].detach()
        if float(args.q_threshold) > 0.0 and route_mode != "oracle":
            q = q * (q >= float(args.q_threshold)).to(q.dtype)
        rec = ct.satr(X_t=x_t, X_aux=x_aux, q=q,
                      neighbour_index=ct.neighbour_index, H_bar=H,
                      bp_messages=diag.get("bp_messages"), syndrome=diag["s"],
                      template_pool=template, memory_readout=None,
                      motion_map=None, topk=int(args.topk))
        out = rec["X_rec"]
        d_in = (1 - F.cosine_similarity(x_t, clean_ref, dim=-1)).clamp(0, 2)
        d_out = (1 - F.cosine_similarity(out, clean_ref, dim=-1)).clamp(0, 2)
        gain = d_in[mask].mean() - d_out[mask].mean()
        healthy = (out[~mask] - x_t[~mask]).square().mean()
        with torch.no_grad():
            clean_head = model.head(clean_ref.float())
        rec_head = model.head(out.float())
        track_dist = F.mse_loss(rec_head["score_map"].float(), clean_head["score_map"].float())
        return out, {
            "gain": float(gain.detach()), "d_in": float(d_in[mask].mean()),
            "d_out": float(d_out[mask].mean()), "healthy_drift": float(healthy.detach()),
            "track_dist": float(track_dist.detach()),
        }

    for step in range(int(args.steps)):
        # Train with oracle locations first. Learned q is evaluated independently below.
        losses = []
        rows = []
        for item in train:
            _out, row = run_one(item, "oracle")
            # Re-run to retain the graph for the loss; diagnosis remains detached.
            _name, x_t, x_aux, template, mask, clean_ref = item
            with torch.no_grad():
                diag = ct.diagnosis(x_t, x_aux, H, template_context=template)
            train_q = mask.float() if args.train_route == "oracle" else diag["q"].detach()
            if float(args.q_threshold) > 0.0 and args.train_route != "oracle":
                train_q = train_q * (train_q >= float(args.q_threshold)).to(train_q.dtype)
            rec = ct.satr(X_t=x_t, X_aux=x_aux, q=train_q,
                          neighbour_index=ct.neighbour_index, H_bar=H,
                          bp_messages=diag.get("bp_messages"), syndrome=diag["s"],
                          template_pool=template, memory_readout=None,
                          motion_map=None, topk=int(args.topk))
            d_out = (1 - F.cosine_similarity(rec["X_rec"], clean_ref, dim=-1)).clamp(0, 2)
            d_in = (1 - F.cosine_similarity(x_t, clean_ref, dim=-1)).clamp(0, 2).detach()
            l_rec = d_out[mask].mean()
            l_gain = F.relu(0.5 * d_in[mask].mean() - (d_in[mask].mean() - d_out[mask].mean()))
            l_id = (rec["X_rec"][~mask] - x_t[~mask]).square().mean()
            with torch.no_grad():
                clean_head = model.head(clean_ref.float())
            rec_head = model.head(rec["X_rec"].float())
            # Feature recovery is subordinate to the actual tracking representation.  This
            # frozen-head distillation term rejects corrections that reduce cosine distance but
            # move the score map away from the clean target response.
            l_track = F.mse_loss(rec_head["score_map"].float(), clean_head["score_map"].float())
            losses.append(l_rec + l_gain + float(args.identity_weight) * l_id + 0.5 * l_track)
            rows.append(row)
        loss = torch.stack(losses).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(ct.satr.parameters(), 1.0)
        opt.step()
        if step % 25 == 0 or step == int(args.steps) - 1:
            with torch.no_grad():
                oracle_rows = [run_one(item, "oracle")[1] for item in valid]
                learned_rows = [run_one(item, "learned")[1] for item in valid]
                oracle_valid = {k: sum(r[k] for r in oracle_rows) / len(oracle_rows)
                                for k in oracle_rows[0]}
                learned_valid = {k: sum(r[k] for r in learned_rows) / len(learned_rows)
                                 for k in learned_rows[0]}
            row = {"step": step, "loss": float(loss.detach()), "grad": float(grad),
                   "train_gain": float(sum(r["gain"] for r in rows) / len(rows)),
                   "valid_oracle": oracle_valid, "valid_learned": learned_valid}
            history.append(row)
            score = -oracle_valid["track_dist"] + oracle_valid["gain"] - 0.1 * oracle_valid["healthy_drift"]
            if score > best["score"]:
                best = {"score": score, "step": step,
                        "state": {k: v.detach().cpu().clone() for k, v in ct.satr.state_dict().items()}}
            print(json.dumps(row), flush=True)

    if best["state"] is None:
        raise RuntimeError("no finite recovery checkpoint")
    save_file(best["state"], str(args.save_checkpoint))
    ct.satr.load_state_dict(best["state"], strict=True)
    with torch.no_grad():
        final = {}
        for mode in ("oracle", "learned"):
            rows = [run_one(item, mode)[1] for item in valid]
            final[mode] = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}
    report = {"probe": "recovery_stage2_v1", "seed": args.seed, "steps": args.steps,
              "lr": args.lr, "topk": args.topk, "diagnosis_checkpoint": str(args.diagnosis_checkpoint),
              "best_step": best["step"], "best_score": best["score"],
              "final_valid": final, "history": history,
              "checkpoint": str(args.save_checkpoint)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
