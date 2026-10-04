"""Apply fix #3 -- make the motion prior's attention bias actually discriminative.

MEASURED DEFECT (2026-10-04):
    motion_map_norm   absmax 8.34e-03   std 2.03e-03   mean 3.91e-03
The map is normalised to unit mass over 16x16=256 cells, so every cell sits at ~1/256 = 3.9e-3
and the spread is only +-2e-3.  The refiner turns it into an attention bias as

    attn_bias = log(clamp(motion_map, min=1e-4)) * motion_bias_scale

With the map clustered at 3.9e-3, ``log`` lands in about [-1.6, -1.5]: a total range of ~0.1
nats, times scale 0.5 gives a 0.05 logit spread across the 8 neighbours.  A softmax over 8
entries with a 0.05 spread is numerically indistinguishable from a uniform softmax, so the
Kalman prior has NO effect on evidence routing (measured ablation: zeroing ``motion_map``
changes ``X_rec`` by 7.4e-07 relative).

THE FIX: normalise the log-bias to zero mean and unit scale before applying
``motion_bias_scale``.  This makes ``motion_bias_scale`` mean what its name says -- "how many
nats of spread does the prior contribute" -- instead of depending on the arbitrary offset of
a unit-mass map.  The map's *shape* (where the mass is) is preserved; only the constant offset
is removed.

    bias = (log m - mean(log m)) / std(log m) * motion_bias_scale

Backwards compatibility: with ``motion_bias_normalise=False`` the old expression is used
verbatim, so a checkpoint trained before this change loads and behaves identically.
"""
R = "/home/yangjuanfeng/lab/projects/gola-CodeTrack/codetrack/recovery.py"
src = open(R).read()
orig = src

old = """        # ---- 2. motion prior enters as an attention bias ------------------------
        attn_bias = None
        if motion_map is not None:
            grid = int(round(math.sqrt(n)))
            flat_map = motion_map.reshape(motion_map.shape[0], -1)    # (B, N)
            # log-space bias, scaled small so it nudges rather than dominates
            nb_bias = flat_map[bidx[:, :, None], nb]                  # (B, K, K_n)
            attn_bias = torch.log(nb_bias.clamp(min=1e-4)) * float(self.motion_bias_scale)"""

new = """        # ---- 2. motion prior enters as an attention bias ------------------------
        attn_bias = None
        if motion_map is not None:
            flat_map = motion_map.reshape(motion_map.shape[0], -1)    # (B, N)
            nb_bias = flat_map[bidx[:, :, None], nb]                  # (B, K, K_n)
            log_b = torch.log(nb_bias.clamp(min=1e-4))
            # REVISED 2026-10-04.  ``motion_map`` is normalised to unit mass, so its entries
            # cluster at 1/256 = 3.9e-3 with a spread of only 2e-3.  ``log`` of that is a
            # constant -1.55 plus +-0.5, and scaling it down gave a logit spread of ~0.05 nats
            # over the 8 neighbours -- a softmax that is numerically uniform, which is why
            # zeroing ``motion_map`` changed ``X_rec`` by only 7.4e-07 relative.
            # Centring and standardising the log-bias makes ``motion_bias_scale`` a real
            # "nats of spread" knob independent of the map's arbitrary offset, while preserving
            # the prior's SHAPE (which cells are favoured).  Set
            # ``motion_bias_normalise=False`` to reproduce the pre-2026-10-04 numerics exactly.
            if self.motion_bias_normalise:
                dims = tuple(range(1, log_b.dim()))
                mu = log_b.mean(dim=dims, keepdim=True)
                sd = log_b.std(dim=dims, keepdim=True, unbiased=False).clamp(min=1e-6)
                log_b = (log_b - mu) / sd
            attn_bias = log_b * float(self.motion_bias_scale)"""

assert old in src, "motion bias anchor not found"
src = src.replace(old, new, 1)

old_sig = """                 residual_gate_init: float = -8.0, motion_bias_scale: float = 0.5,
                 up_init_std: float = 0.02):"""
new_sig = """                 residual_gate_init: float = -8.0, motion_bias_scale: float = 0.5,
                 up_init_std: float = 0.02, motion_bias_normalise: bool = True):"""
assert old_sig in src, "refiner signature anchor 2 not found"
src = src.replace(old_sig, new_sig, 1)

old_attr = """        self.memory_dim = memory_dim
        self.motion_bias_scale = float(motion_bias_scale)

        # condition = [ K_n neighbours (flattened) | X_aux_i | template | memory | q_i ]"""
new_attr = """        self.memory_dim = memory_dim
        self.motion_bias_scale = float(motion_bias_scale)
        self.motion_bias_normalise = bool(motion_bias_normalise)

        # condition = [ K_n neighbours (flattened) | X_aux_i | template | memory | q_i ]"""
assert old_attr in src, "refiner attr anchor not found"
src = src.replace(old_attr, new_attr, 1)

open(R, "w").write(src)
print("recovery.py motion-bias patched: %d -> %d bytes" % (len(orig), len(src)))
