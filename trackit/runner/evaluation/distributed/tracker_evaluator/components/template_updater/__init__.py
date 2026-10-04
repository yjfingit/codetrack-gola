# Added by Zekai Shao
# Licensed under Apache-2.0: http://www.apache.org/licenses/LICENSE-2.0
# Add type hint

from typing import Tuple
import numpy as np
import torch


class TemplateUpdater:
    def __init__(self):
        ...

    def start(self, max_batch_size: int, template_shape: Tuple[int, int, int]):
        ...

    def stop(self):
        ...

    def initialize(self, task_id, template: torch.Tensor):
        ...

    def update(self, task_id, confidence: float, x: torch.Tensor, bbox: np.ndarray,
               quality: float = 1.0):
        """``quality`` in [0, 1] is CodeTrack's block-6 template-protection signal.

        The architecture figure makes template updating conditional on
        ``score_t > 0.84 AND c_t > tau_c``.  The score half is enforced by the caller's
        ``confidence``; ``quality`` carries the second half.  Defaulting to 1.0 keeps every
        upstream updater (and any config that does not opt in) behaving exactly as before.
        """
        ...

    def delete(self, task_id) -> None:
        ...

    def get(self, task_id) -> torch.Tensor:
        ...

    def get_batch(self, task_ids: list) -> torch.Tensor:
        ...
