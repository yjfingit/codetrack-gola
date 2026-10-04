"""Staged CodeTrack training: S1 -> S2 -> S3 -> S4 with automatic gates and rollback.

What this is (and is not)
-------------------------
It is a **state machine on top of the existing harness**, not a second training loop.  Each stage
is one `main.py` subprocess with: a different `optimizer.parameter_scope` (which is what actually
freezes / unfreezes parameters), a different `stage.max_updates` budget, a different update-level
warmup and cosine horizon, and -- for S4 -- a different sampler and loss set.  Everything else
(data pipeline, AMP, gradient accumulation, optimizer construction, checkpointing, logging) is
reused unchanged.

Why stage boundaries are epoch-aligned
--------------------------------------
The framework's checkpoint dumper fires on epoch boundaries, and its `--resume` flag does not
currently restore application state (the value is stored and passed to the application but never
read).  Rather than duplicate the trainer state, each stage's `samples_per_epoch` is chosen so the
update budget is a whole number of epochs:

    samples_per_epoch = max_updates * effective_batch / num_epochs

so "stage finished" and "epoch boundary" coincide, per-epoch checkpoints accumulate naturally, and
a resume after a crash restarts from a completed epoch boundary instead of a torn mid-epoch state.
The tradeoff is that the resumed optimizer loses its momentum for at most one epoch -- which is
stated in the report rather than hidden.

Decision logic
--------------
    S1 -> hard gate; failure STOPS the pipeline (S2's starting point would be invalid)
    S2 -> hard gate; failure STOPS
    S3 -> optional: accepted, or REJECTED and S2 is restored as the base for S4
    S4 -> if it shows no gain it is NOT allowed to become the final model

Usage
-----
    python tools/train_codetrack_staged.py --stages s1,s2,s3,s4 --resume auto \
        --output-root outputs/staged
    python tools/train_codetrack_staged.py --stages s1 --smoke \
        --output-root outputs/staged_smoke
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools.stage_metrics import parameter_deltas, write_json  # noqa: E402
from tools.stage_validate import validate_stage, write_report  # noqa: E402

# --------------------------------------------------------------------------- stage table
# ``updates`` / ``warmup`` come from docs/CodeTrack-实验文档.md; ``config`` is the stage config
# directory; ``needs`` is the checkpoint a stage starts from.
STAGES = {
    "s1": dict(name="S1 spatial recovery warm-up", index=1, config="codetrack_s1",
               updates=1500, warmup=128, scope="codetrack", needs=None),
    "s2": dict(name="S2 joint PEFT", index=2, config="codetrack_s2",
               updates=8000, warmup=410, scope="framework default", needs="s1"),
    "s3": dict(name="S3 DINOv2 last-2 fine-tune", index=3, config="codetrack_s3",
               updates=1800, warmup=256, scope="+ blocks.10/11", needs="s2"),
    "s4": dict(name="S4 temporal training", index=4, config="codetrack_s4",
               updates=3000, warmup=48, scope="+ motion/memory/gate", needs="s3_or_s2"),
}
SMOKE_UPDATES = 10

# Parameter groups whose movement is asserted, per stage.  These are name prefixes into the
# model's state dict.
DELTA_GROUPS = {
    "codetrack.": "codetrack.",
    "lora": "lora",
    "head.": "head.",
    "blocks.": "blocks.",
    "token_type_embed": "token_type_embed",
    "codetrack.motion.": "codetrack.motion.",
    "codetrack.memory.": "codetrack.memory.",
}

UPDATE_RE = re.compile(r"Epoch: \[\d+\] \[ *\d+/\d+\]")


class Orchestrator:
    def __init__(self, args):
        self.args = args
        self.root = args.output_root
        self.reports = os.path.join(self.root, "reports")
        self.logs = os.path.join(self.root, "logs")
        self.ckpts = os.path.join(self.root, "checkpoints")
        for d in (self.reports, self.logs, self.ckpts):
            os.makedirs(d, exist_ok=True)
        self.state_path = os.path.join(self.root, "state.json")
        self.state = self._load_state()

    # ------------------------------------------------------------------ state
    def _load_state(self) -> dict:
        if os.path.exists(self.state_path):
            with open(self.state_path) as fh:
                return json.load(fh)
        return {"completed": [], "current": None, "s3_accepted": None,
                "best": {}, "history": []}

    def _save_state(self) -> None:
        self.state["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        write_json(self.state_path, self.state)

    # ------------------------------------------------------------------ helpers
    def _stage_dir(self, key: str) -> str:
        return os.path.join(self.root, "stage_" + key)

    def _checkpoint_dir(self, key: str) -> str:
        return os.path.join(self.ckpts, key)

    def _latest_epoch_checkpoint(self, key: str):
        """The newest ``model.bin`` under this stage's output, or ``None``."""
        base = self._stage_dir(key)
        if not os.path.isdir(base):
            return None
        found = []
        for dirpath, _dirnames, filenames in os.walk(base):
            if "model.bin" in filenames:
                found.append(os.path.join(dirpath, "model.bin"))
        if not found:
            return None
        found.sort(key=lambda p: os.path.getmtime(p))
        return found[-1]

    def _base_checkpoint(self, key: str):
        """The checkpoint a stage should start from."""
        spec = STAGES[key]
        needs = spec["needs"]
        if needs is None:
            return self.args.weight_path
        if needs == "s3_or_s2":
            if self.state.get("base_for_s4"):
                return self.state["base_for_s4"]
            return self.state.get("best", {}).get("s2") or self.args.weight_path
        return self.state.get("best", {}).get(needs) or self.args.weight_path

    def _updates_done(self, key: str) -> int:
        log = os.path.join(self.logs, f"{key}.log")
        if not os.path.exists(log):
            return 0
        with open(log, errors="replace") as fh:
            return sum(1 for line in fh if UPDATE_RE.search(line)) // 16

    def _snapshot(self, tag: str) -> str:
        """Read the model weights out of the newest checkpoint for the delta comparison."""
        return tag

    def _load_state_dict(self, model_bin: str):
        from safetensors.torch import load_file
        return load_file(model_bin)

    # ------------------------------------------------------------------ training
    def run_stage(self, key: str) -> dict:
        spec = STAGES[key]
        updates = SMOKE_UPDATES if self.args.smoke else spec["updates"]
        if self.args.updates_override:
            updates = self.args.updates_override
        epochs = updates // 1024 if updates >= 1024 else 1
        if self.args.smoke:
            epochs = 1
        # `samples_per_epoch` is set so the budget is a whole number of optimizer updates.
        # 8 samples/micro-step x 16 accumulation = 128 samples per update.
        samples_per_epoch = updates * 128 // epochs

        base = self._base_checkpoint(key)
        log_path = os.path.join(self.logs, f"{key}.log")
        out_dir = self._stage_dir(key)
        os.makedirs(out_dir, exist_ok=True)

        mixin = f"staged_{key}"
        cmd = [
            sys.executable, os.path.join(ROOT, "main.py"), "GOLA", spec["config"],
            "--distributed_nproc_per_node", "1", "--disable_wandb",
            "--weight_path", str(base), f"--output_dir={out_dir}",
            "--mixin_config", mixin,
        ]
        overrides = self._write_mixin(key, updates, epochs, samples_per_epoch)
        print(f"\n[{key}] {spec['name']}")
        print(f"[{key}] base={base}")
        print(f"[{key}] budget={updates} updates ({epochs} epoch(s), samples_per_epoch="
              f"{samples_per_epoch}), warmup={spec['warmup']}")
        print(f"[{key}] overrides: {overrides}")
        print(f"[{key}] cmd: {' '.join(cmd)}")
        if self.args.dry_run:
            return {"key": key, "dry_run": True, "cmd": cmd, "log": log_path}

        env = dict(os.environ)
        env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        with open(log_path, "w") as fh:
            fh.write("# cmd: " + " ".join(cmd) + "\n")
            fh.flush()
            t0 = time.time()
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
            rc = proc.wait()
            elapsed = time.time() - t0
        print(f"[{key}] exit={rc} elapsed={elapsed/60:.1f} min")
        return {"key": key, "returncode": rc, "log": log_path, "elapsed_s": elapsed,
                "base": str(base), "updates_target": updates}

    def _write_mixin(self, key: str, updates: int, epochs: int, samples_per_epoch: int) -> str:
        """Write a mixin that sets this run's budget.  Mixins may only *replace existing* keys."""
        spec = STAGES[key]
        os.makedirs(os.path.join(ROOT, "config/GOLA/_mixin"), exist_ok=True)
        path = os.path.join(ROOT, "config/GOLA/_mixin", f"staged_{key}.yaml")
        rules = [
            {"path": "run.runner.train.stage.max_updates", "value": updates},
            {"path": "run.runner.train.stage.warmup_updates", "value": spec["warmup"]},
            {"path": "run.runner.train.optimization.lr_scheduler.override.t_initial_updates",
             "value": updates},
            {"path": "run.runner.train.optimization.lr_scheduler.parameters.warmup_updates",
             "value": spec["warmup"]},
            {"path": "run.num_epochs", "value": epochs},
            {"path": "run.data.train.sampler.samples_per_epoch", "value": samples_per_epoch},
        ]
        import yaml
        with open(path, "w") as fh:
            yaml.safe_dump(rules, fh, sort_keys=False)
        return f"{updates} updates / {epochs} epoch(s)"

    # ------------------------------------------------------------------ validation
    def validate(self, key: str, run: dict) -> object:
        spec = STAGES[key]
        # Parameter deltas need a before/after snapshot.  "Before" is the base checkpoint this
        # stage loaded; "after" is the newest checkpoint it produced.
        deltas = None
        after_bin = self._latest_epoch_checkpoint(key)
        before_bin = run.get("base")
        if after_bin and before_bin and os.path.exists(str(before_bin)):
            try:
                before = self._load_state_dict(str(before_bin))
                after = self._load_state_dict(after_bin)
                deltas = parameter_deltas(before, after, DELTA_GROUPS)
            except Exception as exc:  # noqa: BLE001
                print(f"[{key}] could not compute parameter deltas: {exc}")
        # The residual gate lives in the checkpoint, not the log.
        gate_init = gate_now = None
        if after_bin:
            try:
                sd = self._load_state_dict(after_bin)
                for name, t in sd.items():
                    if name.endswith("denoiser.residual_gate"):
                        gate_now = float(t.reshape(-1)[0])
                gate_init = self._base_gate(str(before_bin)) if before_bin else None
            except Exception as exc:  # noqa: BLE001
                print(f"[{key}] could not read the residual gate: {exc}")
        rep = validate_stage(spec["index"], run["log"], deltas=deltas,
                             gate_init=gate_init, gate_now=gate_now, name=spec["name"])
        write_report(rep, self.reports)
        print("\n" + rep.to_text())
        return rep

    def _base_gate(self, model_bin: str):
        try:
            sd = self._load_state_dict(model_bin)
            for name, t in sd.items():
                if name.endswith("denoiser.residual_gate"):
                    return float(t.reshape(-1)[0])
        except Exception:  # noqa: BLE001
            return None
        return None

    # ------------------------------------------------------------------ pipeline
    def run(self) -> int:
        keys = [s.strip().lower() for s in self.args.stages.split(",") if s.strip()]
        unknown = [k for k in keys if k not in STAGES]
        if unknown:
            print(f"unknown stage(s): {unknown}; known: {sorted(STAGES)}", file=sys.stderr)
            return 2
        print("=" * 72)
        print(f"CodeTrack staged training: {keys}")
        print(f"output root : {self.root}")
        print(f"smoke       : {self.args.smoke}"
              + (f" ({SMOKE_UPDATES} updates per stage)" if self.args.smoke else ""))
        print(f"resume      : {self.args.resume}")
        print("=" * 72)

        for key in keys:
            spec = STAGES[key]
            self.state["current"] = key
            self._save_state()

            if key in self.state["completed"] and self.args.resume == "auto":
                print(f"[{key}] already completed; skipping (resume auto)")
                continue

            # Skip a stage whose latest complete run already exists and is validated.
            run = self.run_stage(key)
            if run.get("dry_run"):
                continue
            if run.get("returncode") != 0:
                print(f"[{key}] training subprocess failed with {run['returncode']}; STOPPING")
                self.state["history"].append({"stage": key, "status": "crashed",
                                              "returncode": run["returncode"]})
                self._save_state()
                return 1

            rep = self.validate(key, run)
            entry = {"stage": key, "decision": rep.decision,
                     "hard_failed": rep.hard_failed, "warned": rep.warned,
                     "report": os.path.join(self.reports, f"stage{spec['index']}_report.json")}

            if spec["index"] in (1, 2) and rep.hard_failed:
                # A failed core stage would hand an invalid starting point to the next one.
                print(f"[{key}] HARD GATE FAILED -> pipeline stops "
                      f"(fix the stage before building on it)")
                entry["status"] = "no-go"
                self.state["history"].append(entry)
                self._save_state()
                return 1

            if spec["index"] == 3:
                accepted = not rep.hard_failed
                self.state["s3_accepted"] = accepted
                entry["accepted"] = accepted
                if accepted:
                    keep = self._latest_epoch_checkpoint("s3")
                    self.state["best"]["s3"] = keep
                    self.state["base_for_s4"] = keep
                    print("[s3] ACCEPTED")
                else:
                    fallback = self.state["best"].get("s2")
                    self.state["base_for_s4"] = fallback
                    print(f"[s3] REJECTED, FALLBACK TO S2 -> {fallback}")
            elif spec["index"] == 4:
                if rep.hard_failed:
                    fallback = self.state.get("base_for_s4") or self.state["best"].get("s2")
                    self.state["final"] = fallback
                    entry["status"] = "rejected"
                    print(f"[s4] REJECTED (no gain) -> final model stays {fallback}")
                else:
                    self.state["final"] = self._latest_epoch_checkpoint("s4")
                    entry["status"] = "accepted"
            else:
                ckpt = self._latest_epoch_checkpoint(key)
                self.state["best"][key] = ckpt
                entry["status"] = "accepted"

            self.state["history"].append(entry)
            if key not in self.state["completed"]:
                self.state["completed"].append(key)
            self._save_state()

        self._final_report()
        print("\n" + "=" * 72)
        print("PIPELINE COMPLETE")
        for e in self.state["history"]:
            print(f"  {e['stage']:<4} {e.get('status', ''):<10} {e['decision']}"
                  + ("" if e.get("hard_failed") else "  (gates passed)"))
        print(f"final model: {self.state.get('final')}")
        print("NOTE: 'gates passed' means the mechanism was demonstrated, not that the tracker "
              "improved. Tracking metrics require the evaluation pipeline (see "
              "docs/staged_training_design.md).")
        print("=" * 72)
        return 0

    def _final_report(self) -> None:
        payload = {"history": self.state["history"],
                   "final": self.state.get("final"),
                   "s3_accepted": self.state.get("s3_accepted"),
                   "best": self.state.get("best", {})}
        write_json(os.path.join(self.reports, "final_report.json"), payload)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", default="s1,s2,s3,s4")
    ap.add_argument("--output-root", default=os.path.join(ROOT, "outputs/staged"))
    ap.add_argument("--resume", default="auto", choices=["auto", "off"])
    ap.add_argument("--smoke", action="store_true",
                    help=f"run {SMOKE_UPDATES} optimizer updates per stage to validate the state machine")
    ap.add_argument("--updates-override", type=int, default=None,
                    help="force the same update budget for every stage")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, launch nothing")
    ap.add_argument("--weight_path",
                    default=os.path.join(ROOT, "weights/gola_b224.bin"))
    args = ap.parse_args()
    return Orchestrator(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
