"""Real corruption/pointer checks, independent of ML dependencies."""
import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("verify_assets", Path(__file__).resolve().parents[2] / "scripts/verify_assets.py")
assets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assets)


class AssetIdentityTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "weights.pth"
        self.payload = b"example serialized model bytes"
        self.pointer = "version https://git-lfs.github.com/spec/v1\noid sha256:" + hashlib.sha256(self.payload).hexdigest() + f"\nsize {len(self.payload)}\n"

    def test_matching_identity(self):
        self.path.write_bytes(self.payload)
        self.assertEqual(assets.verify_file(self.path, self.pointer)["identity"], "PASS")

    def test_same_size_corruption_fails(self):
        self.path.write_bytes(b"X" + self.payload[1:])
        with self.assertRaises(ValueError):
            assets.verify_file(self.path, self.pointer)

    def test_pointer_instead_of_weights_fails(self):
        self.path.write_text(self.pointer)
        with self.assertRaisesRegex(ValueError, "still a pointer"):
            assets.verify_file(self.path, self.pointer)

    def test_missing_weights_fail(self):
        with self.assertRaises(FileNotFoundError):
            assets.verify_file(self.path, self.pointer)

    def test_malformed_pointer_fails(self):
        with self.assertRaises(ValueError):
            assets.parse_pointer("not a pointer")

    def test_invalid_oid_fails(self):
        with self.assertRaises(ValueError):
            assets.parse_pointer(self.pointer.replace("sha256:", "sha512:"))
