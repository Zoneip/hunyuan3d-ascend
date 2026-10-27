#!/usr/bin/env python3
"""Explicit near-white border removal; preserve enclosed white fur and dark parts."""

import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image


def border_alpha(rgb, threshold=237):
    near = (rgb.min(axis=2) > threshold).astype("uint8")
    padded = np.pad(near, 1, constant_values=1)
    cv2.floodFill(padded, None, (0, 0), 2)
    return (padded[1:-1, 1:-1] != 2).astype("uint8") * 255


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--white-background",
        action="store_true",
        help="Explicitly assert that border-connected near-white is background",
    )
    p.add_argument(
        "--split-two",
        action="store_true",
        help="Two largest non-white components, left=front/right=back; inspect results",
    )
    p.add_argument("--threshold", type=int, default=237)
    a = p.parse_args()
    if not 0 <= a.threshold <= 254:
        p.error("threshold must be between 0 and 254")
    if a.output.exists() and any(a.output.iterdir()):
        p.error("Output must be new or empty; source is never overwritten")
    with Image.open(a.input) as im:
        source = np.asarray(im.convert("RGBA"))
    if not a.white_background and source[:, :, 3].min() == 255:
        p.error(
            "Opaque input: explicitly use --white-background, or supply a manually prepared RGBA image"
        )
    if a.split_two and not a.white_background:
        p.error("--split-two currently requires --white-background")
    rgb = source[:, :, :3]
    h, w = rgb.shape[:2]
    boxes = [(0, 0, w, h)]
    names = ["front"]
    if a.split_two:
        _, _, stats, _ = cv2.connectedComponentsWithStats(
            (rgb.min(2) < 235).astype("uint8")
        )
        components = sorted(stats[1:], key=lambda s: int(s[4]), reverse=True)[:2]
        if len(components) != 2 or any(s[4] < 0.01 * w * h for s in components):
            p.error("Could not find two substantial views; crop manually instead")
        components = sorted(components, key=lambda s: int(s[0]))
        if components[0][0] + components[0][2] > components[1][0]:
            p.error("Two component boxes overlap horizontally; crop manually")
        boxes = [
            (
                max(0, int(x) - 20),
                max(0, int(y) - 20),
                min(w, int(x + bw) + 20),
                min(h, int(y + bh) + 20),
            )
            for x, y, bw, bh, _ in components
        ]
        names = ["front", "back"]
    a.output.mkdir(parents=True, exist_ok=True)
    records = []
    for name, (x0, y0, x1, y1) in zip(names, boxes):
        crop = source[y0:y1, x0:x1].copy()
        if a.white_background:
            crop[:, :, 3] = np.minimum(
                crop[:, :, 3], border_alpha(crop[:, :, :3], a.threshold)
            )
        if not crop[:, :, 3].any():
            raise RuntimeError("Empty foreground: check threshold and source")
        Image.fromarray(crop).save(a.output / f"{name}.png")
        Image.fromarray(crop[:, :, 3]).save(a.output / f"{name}_mask.png")
        bg = Image.new("RGBA", (crop.shape[1], crop.shape[0]), "#666666")
        bg.alpha_composite(Image.fromarray(crop))
        bg.convert("RGB").save(a.output / f"{name}_preview.png")
        records.append(
            dict(
                view=name,
                crop=[x0, y0, x1, y1],
                foreground_pixels=int((crop[:, :, 3] > 127).sum()),
            )
        )
    (a.output / "preparation.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    print(json.dumps(records, indent=2))
    print(
        "Inspect *_preview.png before inference; this is not a general-purpose segmentation model."
    )


if __name__ == "__main__":
    main()
