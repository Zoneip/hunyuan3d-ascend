import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import numpy as np
from prepare_images import border_alpha
from weight_checks import align_conditioner_keys, validate_keys
from render_cpu import raster


class WeightChecks(unittest.TestCase):
    def test_known_projection_rename(self):
        old = (
            "main_image_encoder.model.encoder.layer.0.attention.attention.query.weight"
        )
        new = old.replace(".attention.attention.query.", ".attention.q_proj.")
        result, changes = align_conditioner_keys({old: 7}, {new})
        self.assertEqual(result, {new: 7})
        self.assertEqual(len(changes), 1)

    def test_unknown_name_is_not_guessed(self):
        result, changes = align_conditioner_keys(
            {"unexpected.weight": 1}, {"other.weight"}
        )
        self.assertEqual(result, {"unexpected.weight": 1})
        self.assertFalse(changes)

    def test_collision_rejected(self):
        with self.assertRaises(ValueError):
            align_conditioner_keys(
                {
                    "a.attention.attention.query.weight": 1,
                    "a.attention.q_proj.weight": 2,
                },
                {"a.attention.q_proj.weight"},
            )

    def test_only_training_vae_keys_may_be_absent(self):
        self.assertEqual(
            validate_keys("ShapeVAE", ["encoder.input.weight", "pre_kl.bias"], []),
            ["encoder.input.weight", "pre_kl.bias"],
        )
        for name, missing, unexpected in [
            ("ShapeVAE", ["decoder.weight"], []),
            ("ShapeVAE", ["post_kl.weight"], []),
            ("Hunyuan3DDiT", ["encoder.weight"], []),
            ("ShapeVAE", [], ["alien.weight"]),
        ]:
            with self.assertRaises(RuntimeError):
                validate_keys(name, missing, unexpected)


class GeometryChecks(unittest.TestCase):
    def test_white_interior_and_dark_parts_preserved(self):
        rgb = np.full((20, 20, 3), 255, dtype=np.uint8)
        rgb[4:16, 4:16] = [0, 0, 0]
        rgb[7:13, 7:13] = 255
        alpha = border_alpha(rgb)
        self.assertEqual(int(alpha[0, 0]), 0)
        self.assertEqual(int(alpha[10, 10]), 255)
        self.assertEqual(int(alpha[4, 4]), 255)

    def test_complete_raster_and_depth_occlusion(self):
        vertices = np.array(
            [
                [20, 20, 0],
                [120, 20, 0],
                [120, 120, 0],
                [20, 120, 0],
                [30, 30, 1],
                [90, 30, 1],
                [30, 90, 1],
            ],
            dtype=np.float32,
        )
        faces = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6]], dtype=np.int64)
        colors = np.array([[0, 0, 255], [0, 0, 255], [255, 0, 0]], dtype=np.uint8)
        rgba, _ = raster(vertices, faces, colors, 150)
        self.assertEqual(int((rgba[:, :, 3] > 0).sum()), 10000)
        self.assertEqual(rgba[40, 40, :3].tolist(), [255, 0, 0])
        self.assertEqual(rgba[110, 110, :3].tolist(), [0, 0, 255])
        self.assertEqual(int(rgba[0, 0, 3]), 0)

    def test_deep_mesh_stays_inside_every_view(self):
        import tempfile
        import trimesh
        from PIL import Image
        from render_cpu import render

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trimesh.creation.box(extents=[1, 1, 8]).export(root / "deep.obj")
            render(root / "deep.obj", root / "preview", 100)
            for name in ["front", "back", "left", "three_quarter"]:
                alpha = np.asarray(Image.open(root / "preview" / (name + ".png")))[
                    :, :, 3
                ]
                self.assertTrue(alpha.any())
                self.assertFalse(alpha[[0, -1], :].any())
                self.assertFalse(alpha[:, [0, -1]].any())

    def test_repair_refuses_large_secondary_parts(self):
        import tempfile
        import subprocess
        import trimesh

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = trimesh.creation.box()
            second = trimesh.creation.box()
            second.apply_translation([3, 0, 0])
            trimesh.util.concatenate([first, second]).export(root / "multipart.obj")
            script = Path(__file__).resolve().parents[1] / "scripts" / "repair_mesh.py"
            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--mesh",
                    str(root / "multipart.obj"),
                    "--output",
                    str(root / "output"),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("substantial secondary components", result.stderr)
            self.assertFalse((root / "output" / "mesh_editable.glb").exists())


if __name__ == "__main__":
    unittest.main()
