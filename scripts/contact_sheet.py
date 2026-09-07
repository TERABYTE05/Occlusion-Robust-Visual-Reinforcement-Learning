#!/usr/bin/env python
"""Contact sheets for visual verification of camera and occlusion.

Landmine 5 is silent: at 84x84 the block can be a handful of pixels, the vision
branch contributes noise, and it reads as "the method doesn't work". The only
defence is looking at the actual frames, at the actual training resolution,
across *fresh resets* -- the block spawns somewhere new each time, so one reset
proves nothing.

    # is the block visible at all?
    python scripts/contact_sheet.py --n 20 --out results/contact

    # does the mask cover the block wherever it spawns?
    python scripts/contact_sheet.py --n 20 --occlusion static --compare --annotate

Two files are written per run: `*_true.png` is the grid at true 84x84 and is
what you judge; `*_view.png` is upscaled so you can see what you are judging.

``--annotate`` marks the block's projected position. Use it once to confirm the
projection in src/envs/fetch_pixels.py is correct, because tests/test_occlusion.py
trusts that projection to decide whether the mask covers the block.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.envs.fetch_pixels import (  # noqa: E402
    ENV_ID,
    PixelProprioWrapper,
    block_world_position,
    camera_config,
    make_raw_env,
    model_fovy,
    project_world_to_pixel,
)
from src.envs.occlusion import make_occluder  # noqa: E402
from src.utils.gl import print_backend  # noqa: E402


def _grid(frames, cols, pad=2, bg=(24, 24, 28)):
    rows = (len(frames) + cols - 1) // cols
    h, w = frames[0].shape[:2]
    canvas = np.zeros(((h + pad) * rows + pad, (w + pad) * cols + pad, 3), dtype=np.uint8)
    canvas[:] = np.asarray(bg, dtype=np.uint8)
    for i, frame in enumerate(frames):
        r, c = divmod(i, cols)
        y = pad + r * (h + pad)
        x = pad + c * (w + pad)
        canvas[y : y + h, x : x + w] = frame
    return canvas


def _annotate(frame, marks, scale):
    """Upscale a frame and draw block-position markers on it."""
    h, w = frame.shape[:2]
    img = Image.fromarray(frame).resize((w * scale, h * scale), Image.NEAREST)
    if marks:
        draw = ImageDraw.Draw(img)
        for col, row in marks:
            x, y = col * scale, row * scale
            r = max(4, int(0.05 * w * scale))
            draw.ellipse([x - r, y - r, x + r, y + r], outline=(255, 40, 40), width=2)
            draw.line([x - r, y, x + r, y], fill=(255, 40, 40), width=1)
            draw.line([x, y - r, x, y + r], fill=(255, 40, 40), width=1)
    return np.asarray(img, dtype=np.uint8)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=20, help="number of fresh resets")
    p.add_argument("--env-id", default=ENV_ID)
    p.add_argument("--camera", default=None)
    p.add_argument("--image-size", type=int, default=84)
    p.add_argument("--occlusion", default="none", help="none|static|episode_static")
    p.add_argument("--compare", action="store_true", help="clean and occluded side by side")
    p.add_argument("--annotate", action="store_true", help="mark the block's projected position")
    p.add_argument("--warmup", type=int, default=0, help="random steps after each reset")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cols", type=int, default=5)
    p.add_argument("--scale", type=int, default=4, help="upscale factor for the viewing grid")
    p.add_argument("--out", default="results/contact")
    args = p.parse_args()

    print_backend()

    occluder = make_occluder(args.occlusion)
    raw = make_raw_env(env_id=args.env_id, camera=args.camera, render_size=args.image_size)
    # frame_transform is applied manually below so that --compare can show the
    # same underlying frame both ways.
    env = PixelProprioWrapper(raw, image_size=args.image_size, frame_transform=None)

    cfg = camera_config(args.camera)
    fovy = model_fovy(raw)
    rng = np.random.default_rng(args.seed)

    clean, occluded, marks = [], [], []
    covered = 0

    for i in range(args.n):
        env.reset(seed=args.seed + i)
        occluder.reset(raw)
        for _ in range(args.warmup):
            env.step(rng.uniform(raw.action_space.low, raw.action_space.high))

        frame = env.last_frame
        clean.append(frame)
        occluded.append(occluder(frame, raw))

        try:
            col, row = project_world_to_pixel(
                block_world_position(raw), cfg, args.image_size, args.image_size, fovy
            )
        except Exception as exc:  # projection is best-effort; never kill the sheet
            print(f"[contact] reset {i}: projection failed ({exc})")
            marks.append(None)
            continue

        marks.append((col, row))
        inside = occluder.covers(col, row, (args.image_size, args.image_size, 3), raw)
        covered += bool(inside)
        in_frame = 0 <= col < args.image_size and 0 <= row < args.image_size
        flag = "" if in_frame else "  <-- OFF-FRAME: fix the camera"
        print(f"[contact] reset {i:2d}: block at pixel ({col:6.1f},{row:6.1f}) "
              f"occluded={'yes' if inside else 'no '}{flag}")

    env.close()

    if args.occlusion != "none":
        print(f"\n[contact] mask covered the block on {covered}/{args.n} fresh resets")
        if covered < args.n:
            print("[contact] Not all resets are covered. Widen or recentre "
                  "DEFAULT_STATIC_RECT in src/envs/occlusion.py, or retarget the camera.")

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.occlusion}_{args.image_size}px"

    if args.compare and args.occlusion != "none":
        true_frames = [f for pair in zip(clean, occluded) for f in pair]
        view_marks = [m for m in marks for _ in range(2)]
        cols = 2 * max(1, args.cols // 2)
    else:
        true_frames = occluded if args.occlusion != "none" else clean
        view_marks = marks
        cols = args.cols

    true_path = out / f"{tag}_true.png"
    Image.fromarray(_grid(true_frames, cols)).save(true_path)

    view_frames = [
        _annotate(f, [m] if (args.annotate and m) else [], args.scale)
        for f, m in zip(true_frames, view_marks)
    ]
    view_path = out / f"{tag}_view.png"
    Image.fromarray(_grid(view_frames, cols, pad=2 * args.scale)).save(view_path)

    print(f"\n[contact] true-size grid : {true_path}")
    print(f"[contact] viewing grid   : {view_path}")
    print("[contact] Judge visibility from the TRUE-size grid. If the block is "
          "not obvious there, the encoder has no signal (Landmine 5).")


if __name__ == "__main__":
    main()
