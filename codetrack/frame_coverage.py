"""Deterministic, auditable coverage of every search frame in training videos.

Frame zero is initialization only. Invalid annotations stay in temporal order as
absent-target examples. A clip initialization uses only a valid *past* annotation.
Tail windows overlap rather than padding supervised frames. Future-only context may
repeat the last image, but those positions are masked out of utility supervision.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np


@dataclass
class SequenceMetadata:
    name: str
    boxes_xywh: np.ndarray
    visible: tuple[Path, ...]
    infrared: tuple[Path, ...]
    annotation_sha256: str

    @property
    def valid(self):
        return np.isfinite(self.boxes_xywh).all(1) & (self.boxes_xywh[:, 2:] > 0).all(1)


def read_metadata(root, names):
    sequences = []
    for name in names:
        folder = Path(root) / 'trainingset' / name
        annotation = folder / 'init.txt'
        boxes = np.loadtxt(annotation, delimiter=',', ndmin=2)
        visible = tuple(sorted((folder / 'visible').glob('*.jpg')))
        infrared = tuple(sorted((folder / 'infrared').glob('*.jpg')))
        if len(boxes) != len(visible) or len(boxes) != len(infrared):
            raise ValueError(f'Frame/annotation count mismatch: {name}')
        metadata = SequenceMetadata(name, boxes, visible, infrared,
                                    hashlib.sha256(annotation.read_bytes()).hexdigest())
        if len(boxes) < 2 or not metadata.valid[0]:
            raise ValueError(f'{name}: expected a valid frame-zero initialization and >=2 frames')
        sequences.append(metadata)
    return sequences


def target_starts(frame_count, clip_length):
    """Windows cover exactly [1, frame_count); only the final window may overlap."""
    if clip_length < 1 or frame_count < clip_length + 1:
        raise ValueError('Video is shorter than one initialization plus a full clip')
    starts = list(range(1, frame_count - clip_length + 1, clip_length))
    tail = frame_count - clip_length
    if starts[-1] != tail:
        starts.append(tail)
    return starts


def build_plan(sequences, clip_length):
    # Columns: sequence_index, target_start, initialization_frame, clip_length.
    rows, reports = [], []
    for index, metadata in enumerate(sequences):
        n = len(metadata.boxes_xywh)
        valid = metadata.valid
        previous_valid = np.maximum.accumulate(np.where(valid, np.arange(n), -1))
        starts = target_starts(n, clip_length)
        difference = np.zeros(n + 1, dtype=np.int64)
        maximum_gap = 0
        for start in starts:
            initial = int(previous_valid[start - 1])
            if initial < 0 or not valid[initial] or initial >= start:
                raise AssertionError('Initialization must be valid and strictly before prediction')
            rows.append((index, start, initial, clip_length))
            difference[start] += 1
            difference[start + clip_length] -= 1
            maximum_gap = max(maximum_gap, start - initial)
        covered = np.cumsum(difference)[:n]
        if covered[0] != 0 or not (covered[1:] >= 1).all():
            raise AssertionError(f'Incomplete search-frame coverage: {metadata.name}')
        reports.append({
            'name': metadata.name, 'frames': n, 'initialization_only_frames': 1,
            'search_frames': n - 1, 'unique_search_frames_covered': int((covered[1:] > 0).sum()),
            'supervised_frame_presentations': len(starts) * clip_length,
            'tail_overlap_presentations': len(starts) * clip_length - (n - 1),
            'invalid_search_annotations': int((~valid[1:]).sum()),
            'valid_search_frames_covered': int(valid[1:].sum()),
            'windows': len(starts), 'max_initialization_gap_frames': maximum_gap,
            'incomplete_future_utility_targets': min(3, n - 1),
            'annotation_sha256': metadata.annotation_sha256,
        })
    entries = np.asarray(rows, dtype=np.int64).reshape(-1, 4)
    fingerprint = hashlib.sha256(entries.tobytes() + json.dumps(
        [(m.name, m.annotation_sha256) for m in sequences], separators=(',', ':')).encode()).hexdigest()
    report = {
        'sampling': 'frame_coverage_v1', 'clip_length': clip_length,
        'sequences': len(sequences), 'windows': len(entries), 'plan_sha256': fingerprint,
        'search_frames': sum(r['search_frames'] for r in reports),
        'unique_search_frames_covered': sum(r['unique_search_frames_covered'] for r in reports),
        'supervised_frame_presentations': sum(r['supervised_frame_presentations'] for r in reports),
        'invalid_search_annotations': sum(r['invalid_search_annotations'] for r in reports),
        'coverage_fraction': 1.0, 'per_sequence': reports,
        'scope': 'planned full-image search targets; does not guarantee the target lies in predicted crops',
    }
    return entries, report


def shard_plan(entries, epoch, rank, world_size, local_batch, seed=42):
    """Shuffle once globally, pad once globally, then partition without missing rows."""
    if world_size < 1 or not 0 <= rank < world_size or local_batch < 1 or not len(entries):
        raise ValueError('Invalid plan/world size/rank/batch')
    ids = np.random.default_rng(seed + epoch).permutation(len(entries))
    total = math.ceil(len(ids) / (world_size * local_batch)) * world_size * local_batch
    padding = total - len(ids)
    if padding:
        ids = np.concatenate((ids, np.resize(ids, padding)))
    return ids[rank::world_size], padding


def indices_for(entry, frame_count):
    _, start, initial, length = map(int, entry)
    # Three additional images exist only to form four-frame training utility labels.
    requested = np.arange(start, start + length + (3 if length > 1 else 0))
    future_available = requested < frame_count
    indices = [initial, *np.minimum(requested, frame_count - 1).tolist()]
    return indices, future_available
