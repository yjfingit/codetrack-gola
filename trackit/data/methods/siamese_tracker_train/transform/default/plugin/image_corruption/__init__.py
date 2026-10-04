"""Image-level corruption plugin (architecture figure block 1).

The architecture figure attacks the *input*, and the plan's three-tier validation protocol
treats image-level corruption as the stand-in for the "natural" challenge (low light, thermal
crossover, occlusion).  The model previously only applied token-level corruption, so the
diagnosis head would only ever have learned to spot patch erasure.

Where this hooks in
-------------------
``SiamTrackerTrainingPairProcessor.__call__`` runs ``additional_processors`` **after**
``image_normalize_transform_``, so ``collated.input['*_cropped_image']`` is a normalised
float tensor of shape ``(B, 6, H, W)``.  ``codetrack.corruption.corrupt_image`` is written
for exactly that representation (it de-normalises internally), so the plugin operates on the
collated batch rather than re-implementing any pixel maths.

Only the **search region is attacked**.  The template comes from the ground-truth initial
frame, so corrupting it would poison the reference the diagnosis compares against instead of
testing the tracker's ability to notice that its *evidence* is bad.
"""

from typing import Mapping, Sequence

import torch

from trackit.data.protocol.train_input import TrainData

from codetrack.corruption import CorruptionSchedule, corrupt_image


class ImageCorruptionDataCollector:
    """Corrupt the collated search crop and expose the per-channel mask to the model."""

    # The batch collator renames the processor's ``*_cropped_image`` context entries to the
    # model's short input names (``z``/``x``/``d``), so the collated batch is keyed ``x`` --
    # not ``x_cropped_image``.  Verified by instrumenting the collator.
    BATCH_KEY = "x"

    def __init__(self, image_prob: float = 0.45, severity: float = 0.4,
                 enabled: bool = True, seed: int = 0, target: str = "x"):
        self.schedule = CorruptionSchedule(image_prob=image_prob, token_ratio=0.0,
                                           severity=severity, enabled=enabled, seed=seed)
        self.target = target

    def __call__(self, batch: Sequence[Mapping], collated: TrainData) -> None:
        if not self.schedule.enabled:
            return
        key = self.BATCH_KEY
        x = collated.input.get(key)
        if not torch.is_tensor(x) or x.dim() != 4:
            return

        kind, modality, sev, apply = self.schedule.draw_image(x.shape[0], x.device)
        if kind is None or not bool(apply.any()):
            # keep the key present but all-zero, so the model-side branch stays shape-stable
            collated.input["image_corruption_mask"] = torch.zeros(
                x.shape[0], x.shape[1], device=x.device, dtype=torch.float32)
            return

        x_cor, chan_mask = corrupt_image(x, kind, sev, modality=modality)
        # only the sampled rows are replaced; the rest stay clean, which is what keeps clean
        # frames in every batch and stops the diagnosis head collapsing to "always corrupt"
        sel = apply.to(x.device).view(-1, 1, 1, 1)
        collated.input[key] = torch.where(sel, x_cor.to(x.dtype), x)

        # per-sample x per-channel mask = what the model is told was damaged
        mask = chan_mask * apply.to(chan_mask.device, chan_mask.dtype).unsqueeze(-1)
        collated.input["image_corruption_mask"] = mask.to(torch.float32)
        collated.input["image_corruption_kind"] = kind


def build_image_corruption_collector(config: dict) -> ImageCorruptionDataCollector:
    """Read the same knobs the model uses, so the two cannot silently disagree."""
    model_cfg = config.get('model', {})
    ct_cfg = model_cfg.get('codetrack', {}) or {}
    return ImageCorruptionDataCollector(
        image_prob=float(ct_cfg.get('corruption_image_prob', 0.45)),
        severity=float(ct_cfg.get('corruption_severity', 0.4)),
        enabled=bool(ct_cfg.get('corruption_enabled', True)),
        seed=int(ct_cfg.get('corruption_image_seed', 1234)),
        target="x",
    )
