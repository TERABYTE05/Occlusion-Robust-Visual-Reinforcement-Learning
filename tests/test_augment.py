"""Random-shift augmentation, and above all that one shift covers a whole stack."""

import pytest
import torch

from src.models.augment import DEFAULT_PAD, RandomShiftAug

H = 16
STACK = 3


def ramp_batch(batch=4):
    """Frames with distinct values everywhere, so a shift is detectable."""
    row = torch.arange(H, dtype=torch.float32).unsqueeze(1).repeat(1, H)
    col = torch.arange(H, dtype=torch.float32).unsqueeze(0).repeat(H, 1)
    frame = row * H + col
    return frame.expand(batch, 3 * STACK, H, H).clone()


def test_pad_default_is_the_paper_value():
    assert DEFAULT_PAD == 4
    assert RandomShiftAug().pad == 4


def test_output_shape_matches_input():
    x = ramp_batch()
    assert RandomShiftAug(pad=4)(x).shape == x.shape


def test_every_frame_in_a_stack_gets_the_same_shift():
    """The invariant. Independent shifts would inject fake motion into the stack."""
    torch.manual_seed(0)
    x = ramp_batch(batch=8)
    out = RandomShiftAug(pad=4)(x)
    first, second, third = out[:, 0:3], out[:, 3:6], out[:, 6:9]
    assert torch.allclose(first, second, atol=1e-5)
    assert torch.allclose(second, third, atol=1e-5)


def test_different_samples_get_different_shifts():
    torch.manual_seed(0)
    out = RandomShiftAug(pad=4)(ramp_batch(batch=32))
    corners = out[:, 0, 0, 0]
    assert corners.unique().numel() > 1, "the whole batch received one shift"


def test_shift_is_an_integer_crop_within_the_padding():
    """Every output must be exactly one of the 2*pad+1 squared possible crops."""
    pad = 4
    torch.manual_seed(0)
    x = ramp_batch(batch=16)
    out = RandomShiftAug(pad=pad)(x)

    padded = torch.nn.functional.pad(x, (pad,) * 4, mode="replicate")
    crops = [
        padded[:1, :, dy : dy + H, dx : dx + H]
        for dy in range(2 * pad + 1)
        for dx in range(2 * pad + 1)
    ]
    for sample in range(x.shape[0]):
        assert any(
            torch.allclose(out[sample : sample + 1], crop, atol=1e-3) for crop in crops
        ), f"sample {sample} is not an integer crop within +-{pad} px"


def test_a_constant_image_is_unchanged():
    """Replicate padding must not leak a black border into the crop."""
    x = torch.full((4, 3 * STACK, H, H), 7.0)
    assert torch.allclose(RandomShiftAug(pad=4)(x), x, atol=1e-5)


def test_augmentation_is_reproducible_under_a_seed():
    x = ramp_batch(batch=8)
    aug = RandomShiftAug(pad=4)
    torch.manual_seed(123)
    first = aug(x)
    torch.manual_seed(123)
    assert torch.allclose(first, aug(x))


def test_zero_padding_is_the_identity():
    x = ramp_batch()
    assert torch.allclose(RandomShiftAug(pad=0)(x), x, atol=1e-4)


def test_rejects_non_square_frames():
    with pytest.raises(ValueError):
        RandomShiftAug()(torch.zeros(2, 9, 16, 20))


def test_rejects_an_unbatched_input():
    with pytest.raises(ValueError):
        RandomShiftAug()(torch.zeros(9, 16, 16))
