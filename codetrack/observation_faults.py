"""Physical image-observation impairments, before cropping and the normal backbone.

No token tensors or augmentation masks are returned. Occlusion placement can use
the previous predicted box, which is also available to a real tracker.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

TRAIN_FAULTS = ('rgb_lowlight', 'rgb_motionblur', 'tir_contrast_noise')
HELDOUT_FAULTS = ('rgb_occlusion', 'tir_misalignment')


def degrade_observation(image: torch.Tensor, family: str, severity: float,
                        generator: torch.Generator, previous_box=None) -> torch.Tensor:
    """Input/output: float RGB+TIR [6,H,W], in physical [0,255] intensity units."""
    if image.ndim != 3 or image.shape[0] != 6 or not image.is_floating_point():
        raise ValueError('expected physical RGB-T image [6,H,W]')
    if not 0. <= severity <= 1.:
        raise ValueError('severity must be in [0,1]')
    if family == 'natural' or severity == 0.:
        return image.clone()
    out = image.clone()
    _, h, w = image.shape
    if family == 'rgb_lowlight':
        rgb = (image[:3] / 255.).clamp(0., 1.)
        signal = rgb.pow(1. + severity) * (1. - .85 * severity)
        noise = torch.randn(rgb.shape, device=image.device, generator=generator) * (.002 + .008 * severity)
        out[:3] = (signal + noise).clamp(0., 1.) * 255.
    elif family == 'rgb_motionblur':
        length = 2 * int(3 + 12 * severity) + 1
        kernel = image.new_ones(3, 1, 1, length) / length
        padded = F.pad(image[:3][None], (length // 2, length // 2, 0, 0), mode='replicate')
        out[:3] = F.conv2d(padded, kernel, groups=3)[0]
    elif family == 'tir_contrast_noise':
        tir = image[3:]
        centre = tir.mean((-2, -1), keepdim=True)
        # Most LasHeR TIR files are greyscale expanded to RGB. Shared intensity
        # noise avoids manufacturing an unrealistic colour-channel shortcut.
        noise = torch.randn((1, h, w), device=image.device, generator=generator) * (2. + 12. * severity)
        out[3:] = (centre + (tir - centre) * (1. - .85 * severity) + noise).clamp(0., 255.)
    elif family == 'rgb_occlusion':
        if previous_box is None:
            raise ValueError('occlusion placement requires the previous observed box')
        bb = torch.as_tensor(previous_box).detach().cpu().tolist()
        bw, bh = max(4., bb[2] - bb[0]), max(4., bb[3] - bb[1])
        pw = min(w, max(4, int(bw * (.3 + .45 * severity))))
        ph = min(h, max(4, int(bh * (.3 + .45 * severity))))
        x = min(max(0, int((bb[0] + bb[2] - pw) * .5)), w - pw)
        y = min(max(0, int((bb[1] + bb[3] - ph) * .5)), h - ph)
        sx = int(torch.randint(max(1, w - pw + 1), (), device=image.device, generator=generator))
        sy = int(torch.randint(max(1, h - ph + 1), (), device=image.device, generator=generator))
        # A patch from the scene models an opaque background-coloured occluder,
        # without assigning a false token-error label to its pixel footprint.
        out[:3, y:y + ph, x:x + pw] = image[:3, sy:sy + ph, sx:sx + pw]
    elif family == 'tir_misalignment':
        displacement = 2. + 12. * severity
        theta = image.new_tensor([[[1., 0., 2. * displacement / w],
                                   [0., 1., -displacement / h]]])
        grid = F.affine_grid(theta, (1, 3, h, w), align_corners=False)
        out[3:] = F.grid_sample(image[3:][None], grid, mode='bilinear',
                                padding_mode='border', align_corners=False)[0]
    else:
        raise ValueError(f'unknown observation impairment: {family}')
    return out.clamp(0., 255.)
