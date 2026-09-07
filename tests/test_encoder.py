"""The image encoder architecture is an invariant (CLAUDE.md)."""

import pytest
import torch
import torch.nn as nn

from src.envs.proprio import PROPRIO_OBS_DIM
from src.models.encoders import ImageEncoder, ProprioEncoder, conv_output_size


def test_conv_output_size_at_84():
    assert conv_output_size(84) == 35


def test_repr_dim_matches_drqv2():
    assert ImageEncoder(in_channels=9, image_size=84).repr_dim == 32 * 35 * 35 == 39200


def test_forward_shape():
    enc = ImageEncoder(in_channels=9, image_size=84)
    pixels = torch.randint(0, 256, (2, 9, 84, 84), dtype=torch.uint8)
    assert enc(pixels).shape == (2, enc.repr_dim)


def test_architecture_is_four_3x3_convs_with_stride_2_then_1():
    enc = ImageEncoder(in_channels=9, image_size=84)
    convs = [m for m in enc.convnet if isinstance(m, nn.Conv2d)]
    assert len(convs) == 4
    assert [c.stride for c in convs] == [(2, 2), (1, 1), (1, 1), (1, 1)]
    assert all(c.kernel_size == (3, 3) for c in convs)
    assert all(c.out_channels == 32 for c in convs)


def test_a_norm_and_a_relu_follow_every_conv():
    enc = ImageEncoder(in_channels=9, image_size=84)
    modules = list(enc.convnet)
    convs = [i for i, m in enumerate(modules) if isinstance(m, nn.Conv2d)]
    assert len(convs) == 4
    for i in convs:
        assert isinstance(modules[i + 1], (nn.LayerNorm, nn.GroupNorm))
        assert isinstance(modules[i + 2], nn.ReLU)


def test_group_norm_variant_has_the_same_output_shape():
    a = ImageEncoder(in_channels=9, image_size=84, norm="layer")
    b = ImageEncoder(in_channels=9, image_size=84, norm="group")
    assert a.repr_dim == b.repr_dim


def test_proprio_encoder_shape():
    enc = ProprioEncoder(out_dim=128)
    assert enc(torch.randn(4, PROPRIO_OBS_DIM)).shape == (4, 128)


def test_proprio_encoder_refuses_a_wider_input():
    """Widening g_xi's input is how the Landmine 3 leak gets reintroduced."""
    with pytest.raises(ValueError):
        ProprioEncoder(in_dim=25)
