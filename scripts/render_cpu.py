"""Full-triangle CPU orthographic z-buffer, canonical Hunyuan Y-up.
No face deletion, no GL dependency. Positive camera-space Z is nearer.
"""

import sys, json, math
from pathlib import Path
import numpy as np, trimesh
from PIL import Image
from numba import njit


@njit
def raster(points, faces, colors, n):
    depth = np.full((n, n), -1.0e30, np.float32)
    rgb = np.zeros((n, n, 4), np.uint8)
    for i in range(len(faces)):
        ids = faces[i]
        a = points[ids[0]]
        b = points[ids[1]]
        c = points[ids[2]]
        xmin = max(0, int(math.floor(min(a[0], b[0], c[0]))))
        xmax = min(n - 1, int(math.ceil(max(a[0], b[0], c[0]))))
        ymin = max(0, int(math.floor(min(a[1], b[1], c[1]))))
        ymax = min(n - 1, int(math.ceil(max(a[1], b[1], c[1]))))
        den = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(den) < 1.0e-12:
            continue
        for y in range(ymin, ymax + 1):
            for x in range(xmin, xmax + 1):
                u = (
                    (b[1] - c[1]) * (x + 0.5 - c[0]) + (c[0] - b[0]) * (y + 0.5 - c[1])
                ) / den
                v = (
                    (c[1] - a[1]) * (x + 0.5 - c[0]) + (a[0] - c[0]) * (y + 0.5 - c[1])
                ) / den
                t = 1 - u - v
                if u >= -1.0e-6 and v >= -1.0e-6 and t >= -1.0e-6:
                    z = u * a[2] + v * b[2] + t * c[2]
                    if z > depth[y, x]:
                        depth[y, x] = z
                        for k in range(3):
                            rgb[y, x, k] = colors[i, k]
                        rgb[y, x, 3] = 255
    return rgb, depth


def render(src, out, n=900):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    m = trimesh.load(src, force="mesh", process=False)
    verts = np.asarray(m.vertices)
    faces = np.asarray(m.faces, dtype=np.int64)
    lo, hi = verts.min(0), verts.max(0)
    center = (lo + hi) / 2
    v = verts - center
    # Bounding-box diagonal encloses every vertex under any camera rotation.
    # Using only X/Y would clip objects with a long Z axis in side views.
    span = float(np.linalg.norm(hi - lo)) * 1.15
    if not np.isfinite(span) or span <= 0:
        raise ValueError("Mesh has no finite, non-zero spatial extent")
    normals = np.asarray(m.face_normals)
    for label, angle, elev in [
        ("front", 0, 0),
        ("back", 180, 0),
        ("left", 90, 0),
        ("three_quarter", 35, 12),
    ]:
        a, e = np.deg2rad([angle, elev])
        right = np.array([np.cos(a), 0, -np.sin(a)])
        toward = np.array([np.sin(a) * np.cos(e), np.sin(e), np.cos(a) * np.cos(e)])
        up = np.cross(toward, right)
        cam = np.stack([right, up, toward], axis=1)
        p = v @ cam
        p[:, 0] = p[:, 0] / span * n + n / 2
        p[:, 1] = n / 2 - p[:, 1] / span * n
        # Two-sided diffuse light for robust winding-independent inspection.
        light = cam @ np.array([-0.35, 0.65, 0.7])
        light /= np.linalg.norm(light)
        shade = 0.30 + 0.70 * np.abs(normals @ light)
        colors = np.clip(shade[:, None] * np.array([157, 181, 199]), 0, 255).astype(
            np.uint8
        )
        rgba, depth = raster(p.astype(np.float32), faces, colors, n)
        Image.fromarray(rgba).save(out / (label + ".png"))
        mask = rgba[:, :, 3] > 0
        ys, xs = np.where(mask)
        print(
            label,
            "coverage",
            round(float(mask.mean()), 3),
            "bbox",
            [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
            flush=True,
        )
    (out / "render_manifest.json").write_text(
        json.dumps(
            dict(
                source=str(src),
                faces=len(faces),
                vertices=len(verts),
                bounds=[lo.tolist(), hi.tolist()],
                renderer="CPU full-triangle orthographic z-buffer; Y-up; front +Z",
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--size", type=int, default=900)
    args = parser.parse_args()
    if args.size < 32:
        parser.error("--size must be >=32")
    destination = Path(args.output)
    if destination.exists() and any(destination.iterdir()):
        parser.error("Output must be empty")
    render(args.mesh, args.output, args.size)
