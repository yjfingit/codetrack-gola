"""Per-module CodeTrack learnability gate on the real integrated GOLA graph.

For every new parameterised module, freeze all other parameters, execute the full
GOLA -> CodeTrack -> original head -> CodeTrackCriteria path, take a minimal AdamW step, and report:

* boundary output shapes and finite values;
* number and magnitude of finite, non-zero gradients;
* number and magnitude of parameters that actually changed.

This is deliberately a one-step connectivity test, not a convergence or tracking benchmark.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from codetrack.criteria import CodeTrackCriteria  # noqa: E402
from tools.preflight_acceptance import batch, build, targets  # noqa: E402


MODULES = (
    ("parity_check", "H."),
    ("diagnosis", "diagnosis."),
    ("template_pool", "template_pool."),
    ("motion_prior", "motion."),
    ("temporal_memory", "memory."),
    ("sparse_refiner", "satr."),
    ("template_gate", "template_gate."),
)


def _tensor_leaves(value: Any):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensor_leaves(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _tensor_leaves(item)


def _shape_tree(value: Any):
    if torch.is_tensor(value):
        return list(value.shape)
    if isinstance(value, dict):
        return {key: _shape_tree(item) for key, item in value.items()
                if torch.is_tensor(item) or isinstance(item, (dict, tuple, list))}
    if isinstance(value, (tuple, list)):
        return [_shape_tree(item) for item in value]
    return type(value).__name__


def _all_finite(value: Any) -> bool:
    leaves = [tensor for tensor in _tensor_leaves(value) if tensor.is_floating_point()]
    return bool(leaves) and all(bool(torch.isfinite(tensor).all()) for tensor in leaves)


def _expected_shape_ok(name: str, captured: list[Any], model) -> bool:
    if name == "parity_check":
        return tuple(model.codetrack.H.matrix().shape) == (64, 256)
    if not captured:
        return False
    value = captured[-1]
    if name == "diagnosis":
        return tuple(value["q"].shape) == (2, 256) and tuple(value["s"].shape) == (2, 64)
    if name == "template_pool":
        return tuple(value.shape) == (2, 768)
    if name == "motion_prior":
        return (tuple(value["motion_map"].shape) == (2, 1, 16, 16)
                and tuple(value["uncertainty"].shape) == (2,))
    if name == "temporal_memory":
        return (tuple(value["prior_tokens"].shape) == (2, 8, 128)
                and tuple(value["readout"].shape) == (2, 128)
                and tuple(value["trc_confidence"].shape) == (2, 3))
    if name == "sparse_refiner":
        return (tuple(value["X_rec"].shape) == (2, 256, 768)
                and tuple(value["delta"].shape) == (2, 32, 768))
    if name == "template_gate":
        return tuple(value["c_t"].shape) == (2,)
    return False


def _module_for(name: str, model):
    return {
        "diagnosis": model.codetrack.diagnosis,
        "template_pool": model.codetrack.template_pool,
        "motion_prior": model.codetrack.motion,
        "temporal_memory": model.codetrack.memory,
        "sparse_refiner": model.codetrack.satr,
        "template_gate": model.codetrack.template_gate,
    }.get(name)


def main() -> int:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    torch.manual_seed(11)
    model = build().cuda().train()
    criterion = CodeTrackCriteria().cuda()
    model._corruption_schedule = None
    model.codetrack_cfg.corruption_token_prob = 1.0
    model.codetrack_cfg.corruption_image_prob = 0.0
    data, batch_size = batch(b=2, seed=17)
    gt = torch.tensor([[96.0, 96.0, 48.0, 64.0]] * batch_size, device="cuda")
    image_size = torch.tensor([[224.0, 224.0]] * batch_size, device="cuda")
    target = targets(batch_size)

    # Calibrate syndrome gain exactly once before snapshots are taken.  This is part of the
    # intended training initialisation, not an optimiser update by any module arm.
    with torch.no_grad():
        model.reset_sequence()
        model(**data, gt_box=gt, image_size=image_size)
    assert model.codetrack.diagnosis._syndrome_gain_calibrated

    original_requires_grad = {name: param.requires_grad for name, param in model.named_parameters()}
    rows = []
    for module_name, prefix in MODULES:
        for param in model.parameters():
            param.requires_grad_(False)
        selected = [(name, param) for name, param in model.named_parameters()
                    if name.startswith("codetrack." + prefix)]
        if not selected:
            rows.append({"module": module_name, "passed": False,
                         "error": f"no parameters matched codetrack.{prefix}"})
            continue
        for _, param in selected:
            param.requires_grad_(True)

        before = {name: param.detach().clone() for name, param in selected}
        captured: list[Any] = []
        module = _module_for(module_name, model)
        handle = None
        if module is not None:
            handle = module.register_forward_hook(
                lambda _module, _inputs, output, sink=captured: sink.append(output))

        model.zero_grad(set_to_none=True)
        model.reset_sequence()
        torch.manual_seed(101)
        output = model(**data, gt_box=gt, image_size=image_size)
        criterion_output = criterion(output, target)
        loss = criterion_output.loss
        loss_finite = bool(torch.isfinite(loss))
        loss.backward()
        if handle is not None:
            handle.remove()

        first_step_nonzero = sum(
            param.grad is not None and bool(torch.isfinite(param.grad).all())
            and bool((param.grad != 0).any()) for _, param in selected)
        optimizer = torch.optim.AdamW([param for _, param in selected], lr=args.lr,
                                      weight_decay=0.0)
        optimizer.step()

        # The motion prior deliberately zero-initialises its final geometric head.  On the first
        # step that protects the analytic prior but also blocks gradient into the earlier MLP
        # layers.  A second step must open the whole module; otherwise those layers are dead.
        optimizer_steps = 1
        if module_name == "motion_prior":
            optimizer_steps = 2
            model.zero_grad(set_to_none=True)
            model.reset_sequence()
            torch.manual_seed(101)
            output = model(**data, gt_box=gt, image_size=image_size)
            criterion_output = criterion(output, target)
            loss = criterion_output.loss
            loss_finite = loss_finite and bool(torch.isfinite(loss))
            loss.backward()
            optimizer.step()

        gradients = [(name, param.grad) for name, param in selected if param.grad is not None]
        nonzero_gradients = [(name, grad) for name, grad in gradients
                             if bool(torch.isfinite(grad).all()) and bool((grad != 0).any())]
        grad_l1 = float(sum(grad.detach().float().abs().sum() for _, grad in gradients))
        grad_max = float(max((grad.detach().float().abs().max() for _, grad in gradients),
                             default=torch.tensor(0.0, device="cuda")))
        grads_finite = bool(gradients) and all(bool(torch.isfinite(grad).all())
                                               for _, grad in gradients)

        deltas = {name: (param.detach() - before[name]).float()
                  for name, param in selected}
        changed = [name for name, delta in deltas.items() if bool((delta != 0).any())]
        zero_gradient_names = [name for name, param in selected
                               if param.grad is None or not bool((param.grad != 0).any())]
        unchanged_names = [name for name, _ in selected if name not in changed]
        delta_l2 = float(sum(delta.square().sum() for delta in deltas.values()).sqrt())
        delta_max = float(max((delta.abs().max() for delta in deltas.values()),
                              default=torch.tensor(0.0, device="cuda")))

        if module_name == "parity_check":
            boundary_value = model.codetrack.H.matrix()
            captured_shapes = {"H_bar": list(boundary_value.shape)}
            output_finite = _all_finite(boundary_value)
        else:
            captured_shapes = [_shape_tree(item) for item in captured]
            output_finite = bool(captured) and all(_all_finite(item) for item in captured)
        shape_ok = _expected_shape_ok(module_name, captured, model)
        main_shapes_ok = (tuple(output["score_map"].shape) == (2, 16, 16)
                          and tuple(output["boxes"].shape) == (2, 16, 16, 4))
        main_finite = _all_finite({"score_map": output["score_map"], "boxes": output["boxes"]})
        passed = all((loss_finite, grads_finite,
                      len(nonzero_gradients) == len(selected), len(changed) == len(selected),
                      output_finite, shape_ok, main_shapes_ok, main_finite,
                      delta_l2 > 0.0))
        row = {
            "module": module_name,
            "parameter_tensors": len(selected),
            "gradient_tensors": len(gradients),
            "first_step_nonzero_gradient_tensors": first_step_nonzero,
            "nonzero_gradient_tensors": len(nonzero_gradients),
            "changed_parameter_tensors": len(changed),
            "zero_gradient_parameters": zero_gradient_names,
            "unchanged_parameters": unchanged_names,
            "optimizer_steps": optimizer_steps,
            "loss": float(loss.detach()),
            "grad_l1": grad_l1,
            "grad_max": grad_max,
            "delta_l2": delta_l2,
            "delta_max": delta_max,
            "boundary_shapes": captured_shapes,
            "output_finite": output_finite,
            "shape_ok": shape_ok,
            "main_output_finite": main_finite,
            "main_shapes_ok": main_shapes_ok,
            "passed": passed,
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

        # Each arm starts from the same parameters; only recurrent state and RNG are reset.
        with torch.no_grad():
            for name, param in selected:
                param.copy_(before[name])
        del output, criterion_output, loss, optimizer, before, deltas
        torch.cuda.empty_cache()

    for name, param in model.named_parameters():
        param.requires_grad_(original_requires_grad[name])

    # Non-parameterised injector and full-network boundary checks.
    model.reset_sequence()
    torch.manual_seed(101)
    with torch.no_grad():
        final_output = model(**data, gt_box=gt, image_size=image_size)
    extras = final_output["codetrack_extras"]
    mask = extras.get("corruption_mask")
    injector = {
        "module": "token_corruption_injector",
        "parameter_tensors": 0,
        "mask_shape": None if mask is None else list(mask.shape),
        "corrupted_tokens": 0 if mask is None else int(mask.sum()),
        "main_output_finite": _all_finite(final_output["score_map"]),
        "passed": (mask is not None and tuple(mask.shape) == (2, 256)
                   and bool(mask.any()) and _all_finite(final_output["score_map"])),
    }
    rows.append(injector)
    print(json.dumps(injector, ensure_ascii=False), flush=True)

    summary = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "modules_passed": sum(bool(row["passed"]) for row in rows),
        "modules_total": len(rows),
        "all_passed": all(bool(row["passed"]) for row in rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: summary[key] for key in
                      ("modules_passed", "modules_total", "all_passed")}), flush=True)
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
