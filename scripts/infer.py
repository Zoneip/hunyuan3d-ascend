#!/usr/bin/env python3
"""Ascend geometry inference. Uses upstream code; does not vendor model sources."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
from weight_checks import checked_loading


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--family", choices=["2mv", "2.1"], default="2mv")
    p.add_argument(
        "--source", type=Path, required=True, help="Matching upstream repository root"
    )
    p.add_argument(
        "--models", type=Path, required=True, help="Parent of checkpoint subfolder"
    )
    p.add_argument("--front", type=Path, required=True)
    for name in ["back", "left", "right"]:
        p.add_argument("--" + name, type=Path)
    p.add_argument(
        "--output", type=Path, required=True, help="New, empty result directory"
    )
    p.add_argument("--device", default="npu:0")
    p.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--resolution", type=int, default=320)
    p.add_argument("--guidance", type=float, default=5.0)
    p.add_argument("--chunks", type=int, default=8000)
    a = p.parse_args()
    if a.family == "2.1" and any([a.back, a.left, a.right]):
        p.error("2.1 is single-view. Use --family 2mv for labeled joint conditioning.")
    if a.family == "2mv" and not any([a.back, a.left, a.right]):
        p.error("Provide at least one additional labeled view for 2mv.")
    if a.steps < 1 or a.resolution < 16 or a.chunks < 1:
        p.error("steps/chunks must be positive and resolution must be >=16")
    return a


def fingerprint(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    a = arguments()
    source = a.source.resolve()
    package_root = source if a.family == "2mv" else source / "hy3dshape"
    if not package_root.is_dir():
        raise SystemExit(f"Matching upstream source missing: {package_root}")
    if a.output.exists() and any(a.output.iterdir()):
        raise SystemExit("Refusing to overwrite an existing result directory")
    a.output.mkdir(parents=True, exist_ok=True)
    # Load ONLY the requested upstream package in this process.
    sys.path.insert(0, str(package_root))
    import numpy as np
    import torch

    if a.device.startswith("npu"):
        # Source CANN set_env.sh before this import.
        import torch_npu  # noqa: F401

        if not torch.npu.is_available():
            raise SystemExit("NPU unavailable; no silent CPU fallback for the pipeline")
    from PIL import Image

    if a.family == "2mv":
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline as Pipeline
    else:
        from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline as Pipeline

    images = {}
    inputs = {}
    for name in ["front", "left", "back", "right"]:
        path = getattr(a, name)
        if path is None:
            continue
        with Image.open(path) as im:
            if im.mode != "RGBA":
                raise SystemExit(
                    f"{name}: prepare RGBA with --white-background first; RGB is not a mask"
                )
            im = im.copy()
        alpha = np.asarray(im)[:, :, 3]
        if alpha.min() == 255 or alpha.max() == 0:
            raise SystemExit(
                f"{name}: alpha is fully opaque or empty, check segmentation"
            )
        images[name] = im
        inputs[name] = dict(
            path=str(path.resolve()), sha256=fingerprint(path), size=list(im.size)
        )
    conditioning = images if a.family == "2mv" else images["front"]
    reports = []
    started = time.monotonic()
    with checked_loading(torch, reports):
        pipeline = Pipeline.from_pretrained(
            str(a.models.resolve()),
            subfolder="hunyuan3d-dit-v2-mv"
            if a.family == "2mv"
            else "hunyuan3d-dit-v2-1",
            use_safetensors=a.family == "2mv",
            device=a.device,
            dtype=torch.float16 if a.dtype == "fp16" else torch.float32,
        )
    for module in [pipeline.model, pipeline.vae, pipeline.conditioner]:
        module.eval()
    prepared = pipeline.prepare_image(conditioning)
    dimensions = {
        k: list(v.shape) if isinstance(v, torch.Tensor) else str(v)
        for k, v in prepared.items()
    }
    if a.family == "2mv" and (
        prepared["image"].ndim != 5 or prepared["image"].shape[:2] != (1, len(images))
    ):
        raise RuntimeError(
            f"Expected ONE latent conditioned on {len(images)} views: {dimensions}"
        )
    print("CONDITION", json.dumps(dimensions), flush=True)
    load_seconds = time.monotonic() - started
    started = time.monotonic()
    mesh = pipeline(
        image=conditioning,
        num_inference_steps=a.steps,
        guidance_scale=a.guidance,
        octree_resolution=a.resolution,
        num_chunks=a.chunks,
        generator=torch.Generator(device="cpu").manual_seed(a.seed),
        output_type="trimesh",
    )[0]
    if a.device.startswith("npu"):
        torch.npu.synchronize()
    elapsed = time.monotonic() - started
    if mesh is None or not hasattr(mesh, "vertices"):
        raise RuntimeError(
            "No isosurface mesh extracted; increase sampling steps or inspect the scalar field"
        )
    if not np.isfinite(mesh.vertices).all() or not len(mesh.faces):
        raise RuntimeError(
            "Empty or non-finite mesh; output is not a successful reconstruction"
        )
    mesh.export(a.output / "raw.glb")
    mesh.export(a.output / "raw.obj")
    versions = {}
    for name in [
        "torch",
        "torch-npu",
        "transformers",
        "diffusers",
        "huggingface-hub",
        "trimesh",
    ]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    report = dict(
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
        input_images=inputs,
        condition=dimensions,
        weight_checks=reports,
        versions=versions,
        load_seconds=load_seconds,
        generation_seconds=elapsed,
        vertices=len(mesh.vertices),
        faces=len(mesh.faces),
        bounds=mesh.bounds.tolist(),
        watertight=bool(mesh.is_watertight),
        winding_consistent=bool(mesh.is_winding_consistent),
        component_faces=sorted(
            [len(m.faces) for m in mesh.split(only_watertight=False)], reverse=True
        ),
    )
    (a.output / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(
        "RESULT",
        json.dumps(
            {
                k: report[k]
                for k in ["generation_seconds", "vertices", "faces", "watertight"]
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
