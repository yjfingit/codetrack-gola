"""CodeTrack -- ECC-inspired error diagnosis and recovery on top of GOLA.

Architecture figure (block 2 -> 7):

    2  GOLA encoder        fused F0 (B, 768, 768) -> Norm
    3  Error diagnosis     X_t, X_aux -> H (64x256) -> syndrome s -> q in [0,1]^256
    4  Motion prior        b_{t-1} -> constant-velocity KF -> b_hat_t, P_t -> M_t (16x16)
    5  Selective recovery  TopK(q)=E_t -> H-routed sparse cross-attention -> X_t'
    6  Template protection c_t gates the official score>0.84 update rule
    7  Output              X_t' -> original GOLA anchor-free head

The upstream GOLA code under ``trackit/`` is never edited beyond two additive hooks
(``GOLA_DINOv2`` builds a ``CodeTrack`` instance when the config enables it, and its
forward routes through it).  With ``model.codetrack.enabled = false`` the model is
bit-identical to upstream.
"""

from .config import CodeTrackConfig
from .codetrack import CodeTrack

__all__ = ["CodeTrack", "CodeTrackConfig"]
