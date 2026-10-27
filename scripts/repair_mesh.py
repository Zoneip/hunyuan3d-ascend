#!/usr/bin/env python3
"""Keep the main component, repair topology, and export two geometry LODs."""

import argparse
import json
from pathlib import Path
import numpy as np
import pymeshlab
import trimesh


def describe(mesh):
    return dict(
        vertices=len(mesh.vertices),
        faces=len(mesh.faces),
        watertight=bool(mesh.is_watertight),
        winding_consistent=bool(mesh.is_winding_consistent),
        finite_vertices=bool(np.isfinite(mesh.vertices).all()),
        volume=float(mesh.volume),
        euler=int(mesh.euler_number),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mesh", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--faces", type=int, default=80000)
    p.add_argument(
        "--keep-largest",
        action="store_true",
        help="Explicitly discard even substantial secondary components",
    )
    a = p.parse_args()
    if a.faces < 100:
        p.error("--faces must be >=100")
    if a.output.exists() and any(a.output.iterdir()):
        p.error("Output must be empty; input is never overwritten")
    mesh = trimesh.load(a.mesh, force="mesh", process=True)
    parts = sorted(
        mesh.split(only_watertight=False), key=lambda m: len(m.faces), reverse=True
    )
    if not parts:
        raise RuntimeError("Empty input mesh")
    substantial = [
        len(m.faces)
        for m in parts[1:]
        if len(m.faces) > max(2, len(parts[0].faces) * 0.001)
    ]
    if substantial and not a.keep_largest:
        raise SystemExit(
            f"Multipart mesh has substantial secondary components: {substantial}. "
            "No parts were removed. Inspect them; use --keep-largest only if intended."
        )
    mesh = parts[0]
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    report = dict(
        input_faces=sum(len(m.faces) for m in parts),
        removed_component_faces=[len(m.faces) for m in parts[1:]],
        primary_before=describe(mesh),
        actions=[],
    )
    ms = pymeshlab.MeshSet()
    ms.add_mesh(
        pymeshlab.Mesh(
            vertex_matrix=np.asarray(mesh.vertices, dtype=float),
            face_matrix=np.asarray(mesh.faces, dtype=np.int32),
        )
    )
    for name, kwargs in [
        ("meshing_remove_duplicate_faces", {}),
        ("meshing_remove_null_faces", {}),
        ("meshing_repair_non_manifold_edges", {"method": 0}),
        ("meshing_repair_non_manifold_vertices", {}),
        ("meshing_close_holes", {"maxholesize": 80}),
    ]:
        try:
            ms.apply_filter(name, **kwargs)
            report["actions"].append(dict(filter=name, status="ok"))
        except Exception as error:
            report["actions"].append(
                dict(filter=name, status="failed", error=str(error))
            )
    a.output.mkdir(parents=True, exist_ok=True)
    for level in ["high", "editable"]:
        if level == "editable" and ms.current_mesh().face_number() > a.faces:
            ms.meshing_decimation_quadric_edge_collapse(
                targetfacenum=a.faces,
                preserveboundary=True,
                preservenormal=True,
                preservetopology=True,
            )
        current = ms.current_mesh()
        result = trimesh.Trimesh(
            current.vertex_matrix(), current.face_matrix(), process=True
        )
        result.fix_normals()
        report[level] = describe(result)
        result.export(a.output / f"mesh_{level}.glb")
        result.export(a.output / f"mesh_{level}.obj")
    report["geometry_checks_pass"] = all(
        report[level]["watertight"]
        and report[level]["winding_consistent"]
        and report[level]["finite_vertices"]
        and report[level]["volume"] > 0
        for level in ["high", "editable"]
    )
    (a.output / "repair_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report), flush=True)
    if not report["geometry_checks_pass"]:
        raise SystemExit(
            "Geometry checks did not pass; inspect candidates and repair_report.json"
        )


if __name__ == "__main__":
    main()
