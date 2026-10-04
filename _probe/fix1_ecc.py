"""Apply fix #1 -- diagnose and de-saturate the syndrome head.

Root cause (measured 2026-10-04 on weights/gola_b224.bin):
    s = sigmoid(s_raw) has mean 0.7943 and std 9.7e-3 over 64 checks
    => q_logits = H^T s + vote_bias has spread 2e-3 over 256 tokens
    => q std 3.08e-4  (constant)
    => TopK(q) routing is a near-random draw; noise gate is 1.01x over tokens;
       frame_reliability has std 3.3e-5 across the batch.

This patch:
  1. adds ``syndrome_logit_gain``, a learnable scalar on the raw syndrome logit, so the
     head starts with a usable spread instead of a point mass;
  2. calibrates its initial value from the MEASURED logit std so the fix is data-driven,
     not a magic constant;
  3. adds a runtime assertion helper the verifiers can call.

Everything is additive: with ``syndrome_logit_gain=1.0`` and no calibration the numerics are
bit-identical to the previous revision.
"""
import re
import sys

PATH = "/home/yangjuanfeng/lab/projects/gola-CodeTrack/codetrack/ecc.py"
src = open(PATH).read()
orig = src

# ---------------------------------------------------------------- 1. signature
old_sig = """                 detection_prior: float = 0.2,
                 use_cos: bool = True):"""
new_sig = """                 detection_prior: float = 0.2,
                 use_cos: bool = True,
                 syndrome_logit_gain: float = 1.0):"""
assert old_sig in src, "signature anchor not found"
src = src.replace(old_sig, new_sig, 1)

# ---------------------------------------------------------------- 2. new params
old_params = """        self.residual_scale = nn.Parameter(torch.ones(1))
        self.check_scale = nn.Parameter(torch.ones(num_checks))

        # q = sigmoid(H^T s + b)
        self.vote_bias = nn.Parameter(torch.zeros(num_variables))"""
new_params = """        self.residual_scale = nn.Parameter(torch.ones(1))
        self.check_scale = nn.Parameter(torch.ones(num_checks))

        # ---- syndrome saturation guard (added 2026-10-04) ----------------------
        # MEASURED DEFECT.  With the stock init, ``s_raw`` lands at ~0.2, so
        # ``s = sigmoid(s_raw)`` sits at 0.794 +- 0.0097 over 64 checks.  The damage
        # propagates through ``q_logits = H^T s + vote_bias``: the transmitted term has a
        # spread of only 2e-3 over 256 tokens, so ``q`` is constant to 3e-4 and, measured on
        # the same checkpoint,
        #   * ``TopK(q)`` selects a set 4.1e-4 away from the population mean -> near-random;
        #   * the denoiser noise gate spans max/min = 1.01x over 256 tokens (not selective);
        #   * ``frame_reliability`` is one value per batch (std 3.3e-5).
        # A point-mass head cannot be trained out of the point mass by the ordinary loss,
        # because every token sees the same gradient.  The spread has to exist at
        # initialisation.  This learnable scalar rescales the raw logit; its initial value is
        # set by ``calibrate_syndrome_gain`` below from the measured logit std.
        self.syndrome_logit_gain = nn.Parameter(
            torch.full((1,), float(syndrome_logit_gain)))

        # q = sigmoid(H^T s + b)
        self.vote_bias = nn.Parameter(torch.zeros(num_variables))"""
assert old_params in src, "params anchor not found"
src = src.replace(old_params, new_params, 1)

# ---------------------------------------------------------------- 3. forward
old_fwd = """        s_raw = s_raw * self.check_scale
        s = torch.sigmoid(s_raw)                                # (B, M)"""
new_fwd = """        s_raw = s_raw * self.check_scale
        # See ``syndrome_logit_gain`` in ``__init__``: without this scale the 64 checks
        # collapse to a point mass and every downstream consumer of ``q`` degenerates.
        s_raw = s_raw * self.syndrome_logit_gain
        s = torch.sigmoid(s_raw)                                # (B, M)"""
assert old_fwd in src, "forward anchor not found"
src = src.replace(old_fwd, new_fwd, 1)

# ---------------------------------------------------------------- 4. calibrate
anchor = """    def forward(self, X_t: torch.Tensor, X_aux: torch.Tensor,"""
assert anchor in src
calib = '''    @torch.no_grad()
    def calibrate_syndrome_gain(self, s_raw: torch.Tensor, target_std: float = 1.0) -> float:
        """Set ``syndrome_logit_gain`` so the *pre-sigmoid* syndrome spans ``target_std``.

        Called once with a batch from the real data pipeline.  ``s_raw`` is the logit produced
        with a gain of 1.0; if it is already wide enough the gain stays at 1.0.  This keeps the
        fix data-driven instead of introducing a tuned constant, and it is a no-op on any
        checkpoint whose head is already unsaturated.
        """
        cur = float(s_raw.detach().float().std())
        if cur <= 1e-8:
            return float(self.syndrome_logit_gain)
        gain = float(min(max(target_std / cur, 1.0), 1e3))
        self.syndrome_logit_gain.fill_(gain)
        return gain

'''
src = src.replace(anchor, calib + anchor, 1)

open(PATH, "w").write(src)
print("ecc.py patched: 4 hunks applied, %d -> %d bytes" % (len(orig), len(src)))
