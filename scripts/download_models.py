#!/usr/bin/env python3
"""Download geometry weights with exact size, SHA256 and resumable .part files."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from urllib.parse import quote
import time

FAMILIES = {
    "2mv": ("tencent/Hunyuan3D-2mv", "hunyuan3d-dit-v2-mv", "model.fp16.safetensors"),
    "2.1": ("tencent/Hunyuan3D-2.1", "hunyuan3d-dit-v2-1", "model.fp16.ckpt"),
}


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--family", choices=FAMILIES, default="2mv")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--endpoint", default="https://huggingface.co")
    p.add_argument("--attempts", type=int, default=8)
    a = p.parse_args()
    if not a.endpoint.startswith("https://") or a.attempts < 1:
        p.error("Use an HTTPS endpoint and positive attempt count")
    endpoint = a.endpoint.rstrip("/")
    repo, folder, filename = FAMILIES[a.family]
    meta = json.loads(
        subprocess.check_output(
            [
                "curl",
                "-fLsS",
                "--connect-timeout",
                "30",
                "--max-time",
                "60",
                f"{endpoint}/api/models/{repo}?blobs=true",
            ]
        )
    )
    revision = meta["sha"]
    wanted = {f"{folder}/config.yaml", f"{folder}/{filename}"}
    files = [e for e in meta["siblings"] if e["rfilename"] in wanted]
    if {e["rfilename"] for e in files} != wanted:
        raise RuntimeError("Model metadata lacks required files")
    records = []
    for e in files:
        name = e["rfilename"]
        target = a.output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(target.suffix + ".part")
        expected = e["size"]
        sha = e.get("lfs", {}).get("sha256")
        if not target.exists():
            url = f"{endpoint}/{repo}/resolve/{revision}/{quote(name, safe='/')}"
            for attempt in range(a.attempts):
                subprocess.run(
                    [
                        "curl",
                        "-fL",
                        "-C",
                        "-",
                        "--connect-timeout",
                        "30",
                        "--max-time",
                        "900",
                        "--speed-limit",
                        "1024",
                        "--speed-time",
                        "60",
                        "-o",
                        str(partial),
                        url,
                    ],
                    check=False,
                )
                size = partial.stat().st_size if partial.exists() else 0
                if size > expected:
                    raise RuntimeError(
                        f"Oversized partial file: {partial}; inspect it manually"
                    )
                if size == expected:
                    if sha and digest(partial) != sha:
                        raise RuntimeError(
                            f"SHA256 mismatch: {partial}; no file was promoted"
                        )
                    partial.rename(target)
                    break
                time.sleep(2)
            else:
                raise RuntimeError(
                    f"Incomplete download; retained for resume: {partial}"
                )
        if target.stat().st_size != expected:
            raise RuntimeError(f"Existing target has incorrect size: {target}")
        actual = digest(target)
        if sha and actual != sha:
            raise RuntimeError(f"Existing target SHA256 mismatch: {target}")
        records.append(
            dict(file=name, bytes=expected, sha256=actual, matches_lfs=bool(sha))
        )
        print("VERIFIED", json.dumps(records[-1]), flush=True)
    (a.output / f"download-{a.family}.json").write_text(
        json.dumps(dict(repo=repo, revision=revision, files=records), indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
