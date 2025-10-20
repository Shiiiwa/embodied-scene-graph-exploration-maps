from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.components.encoders.exploration_encoder import MetricSemanticMapV2Encoder


def _make_feature_stack(num_channels: int, height: int, width: int) -> np.ndarray:
    """Utility to build a feature tensor with deterministic base channels."""
    feats = np.zeros((num_channels, height, width), dtype=np.float32)
    if num_channels >= 1:
        feats[0] = 0.5  # p_free baseline
    if num_channels >= 2:
        feats[1] = 0.7  # confidence
    if num_channels >= 3:
        feats[2, 0, 0] = 1.0  # visited
    if num_channels >= 4:
        feats[3] = 0.1  # time since visit
    if num_channels >= 5:
        feats[4, 1:, :] = 1.0  # frontier mask slice
    if num_channels >= 6:
        feats[5] = 0.2  # visit count log norm
    return feats


def test_class_alignment_with_explicit_vocab():
    encoder = MetricSemanticMapV2Encoder(
        output_dim=8,
        target_size=(4, 4),
        include_agent=False,
        class_vocab=["Chair", "Mug"],
    )

    base = _make_feature_stack(6, 4, 4)
    class_heat = np.zeros((3, 4, 4), dtype=np.float32)
    class_heat[0, 0, 1] = 0.55  # Mug channel in entry
    class_heat[1, 2, 2] = 0.33  # Chair channel in entry
    class_heat[2, 3, 3] = 0.91  # Other/noise

    feats = np.concatenate([base, class_heat], axis=0)
    meta = {"class_index": {"Mug": 0, "Chair": 1}}

    aligned_feats, aligned_meta = encoder._parse_entry((feats, meta))

    assert aligned_feats.shape == (encoder.base_channels, 4, 4)
    # Provided vocabulary fixes order: Chair first, Mug second, Other last
    np.testing.assert_allclose(aligned_feats[6, 2, 2], 0.33)
    np.testing.assert_allclose(aligned_feats[7, 0, 1], 0.55)
    np.testing.assert_allclose(aligned_feats[8, 3, 3], 0.91)

    meta_tensor = encoder._meta_to_tensor(aligned_meta)
    assert meta_tensor.shape[0] == encoder.meta_dim
    torch.testing.assert_close(
        meta_tensor[-3:],
        torch.tensor([0.33, 0.55, 0.91], dtype=torch.float32),
    )


def test_deterministic_assignment_without_vocab():
    encoder = MetricSemanticMapV2Encoder(
        output_dim=8,
        target_size=(2, 2),
        include_agent=False,
        base_channels=10,  # 6 base + 4 class channels
    )

    base = _make_feature_stack(6, 2, 2)

    class_heat_a = np.zeros((4, 2, 2), dtype=np.float32)
    class_heat_a[0, 0, 0] = 0.4  # Mug in entry
    class_heat_a[1, 1, 1] = 0.8  # Chair in entry
    entry_a = (np.concatenate([base, class_heat_a], axis=0), {"class_index": {"Mug": 0, "Chair": 1}})

    aligned_a, _ = encoder._parse_entry(entry_a)
    # Alphabetical order => Chair -> slot0, Mug -> slot1
    np.testing.assert_allclose(aligned_a[6, 1, 1], 0.8)
    np.testing.assert_allclose(aligned_a[7, 0, 0], 0.4)

    class_heat_b = np.zeros((4, 2, 2), dtype=np.float32)
    class_heat_b[0, 0, 1] = 1.0  # Apple appears later
    entry_b = (np.concatenate([base, class_heat_b], axis=0), {"class_index": {"Apple": 0}})

    aligned_b, _ = encoder._parse_entry(entry_b)
    # Next available slot receives Apple; remaining channel is "other"
    np.testing.assert_allclose(aligned_b[8, 0, 1], 1.0)

    # Re-processing entry_a should keep deterministic mapping
    aligned_a_repeat, _ = encoder._parse_entry(entry_a)
    np.testing.assert_allclose(aligned_a_repeat[6, 1, 1], 0.8)
    np.testing.assert_allclose(aligned_a_repeat[7, 0, 0], 0.4)

    meta_tensor = encoder._meta_to_tensor({})
    assert meta_tensor.shape[0] == encoder.meta_dim
