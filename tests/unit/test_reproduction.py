"""DOC-03 failures must be actionable and must not turn into network installs."""

import contextlib
import hashlib
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]


def load(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


preflight = load("scripts/check_reproduction.py", "doc03_preflight")
setup = load("setup_environment.py", "doc03_setup")
live = load("src/carla_adapter/environment_check.py", "doc03_live")


class ReproductionTests(unittest.TestCase):
    def test_asset_corruption_pointer_and_missing_fail_without_loading_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asset = root / "model.bin"
            asset.write_bytes(b"weights")
            manifest = {"assets": [{"path": asset.name, "size_bytes": 7,
                                   "sha256": hashlib.sha256(b"weights").hexdigest()}]}
            self.assertEqual(preflight.check_assets(root, manifest)[0]["status"], "PASS")
            for contents, reason in [(b"changed", "SHA-256"), (b"short", "size"),
                                     (b"version https://git-lfs.github.com/spec/v1\n", "pointer")]:
                asset.write_bytes(contents)
                result = preflight.check_assets(root, manifest)[0]
                self.assertEqual(result["status"], "FAIL")
                self.assertIn(reason, result["error"])
            asset.unlink()
            self.assertEqual(preflight.check_assets(root, manifest)[0]["status"], "FAIL")

    def test_recorded_package_subset_preserves_build_suffix_and_missing_is_failure(self):
        def version(name):
            if name == "missing":
                raise preflight.importlib.metadata.PackageNotFoundError(name)
            return "2.14.1+cpu"

        results = preflight.check_packages({"torch": "2.14.1", "wrong": "0.0", "missing": "1"}, version)
        self.assertEqual([x["status"] for x in results], ["PASS", "FAIL", "FAIL"])
        self.assertEqual(results[0]["installed"], "2.14.1+cpu")

    def test_actual_client_and_server_must_both_equal_requested_version(self):
        client = Mock()
        for client_version, server_version, passed in [
            ("0.9.16", "0.9.16", True), ("0.9.16", "0.9.15", False),
            ("0.9.15", "0.9.15", False),
        ]:
            client.get_client_version.return_value = client_version
            client.get_server_version.return_value = server_version
            if passed:
                self.assertEqual(live.check_versions(client, "0.9.16")["server"], "0.9.16")
            else:
                with self.assertRaisesRegex(ValueError, "Expected CARLA"):
                    live.check_versions(client, "0.9.16")

    def invoke_setup(self, arguments):
        with patch.object(setup.sys, "argv", ["setup_environment.py", *arguments]), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return setup.main()

    def test_read_only_path_never_calls_install_or_lfs(self):
        with patch.object(setup, "run", return_value=True) as run, \
                patch.object(setup, "pip_install") as install, \
                patch.object(setup, "pull_lfs_weights") as lfs:
            self.assertEqual(self.invoke_setup(["--offline", "--check-only", "--assets-only"]), 0)
            install.assert_not_called()
            lfs.assert_not_called()
            self.assertIn("--assets-only", run.call_args.args[0])

    def test_offline_rejects_direct_urls_nested_requirements_and_options_before_pip(self):
        with tempfile.TemporaryDirectory() as directory:
            lock = Path(directory) / "lock.txt"
            for entry in ["pkg @ https://example.org/model.whl", "-r online.txt",
                          "--extra-index-url https://example.org", "pkg==1", ""]:
                lock.write_text(entry, encoding="utf-8")
                with patch.object(setup, "pip_install") as install:
                    with self.assertRaises(SystemExit) as error:
                        self.invoke_setup(["--offline", "--wheelhouse", directory, "--lock-file", str(lock)])
                    self.assertEqual(error.exception.code, 2)
                    install.assert_not_called()

    def test_offline_uses_only_local_resolution_and_propagates_install_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            lock = Path(directory) / "lock.txt"
            lock.write_text("pkg==1.0 \\\n    --hash=sha256:" + "a" * 64 + "\n", encoding="utf-8")
            arguments = ["--offline", "--wheelhouse", directory, "--lock-file", str(lock)]
            with patch.object(setup, "pip_install", side_effect=[True, True]) as install, \
                    patch.object(setup, "run", return_value=True), \
                    patch.object(setup, "pull_lfs_weights") as lfs:
                self.assertEqual(self.invoke_setup(arguments), 0)
                first, second = [call.args[0] for call in install.call_args_list]
                self.assertIn("--require-hashes", first)
                self.assertIn("--no-index", first)
                self.assertIn("--no-deps", second)
                self.assertIn("--no-build-isolation", second)
                lfs.assert_not_called()
            with patch.object(setup, "pip_install", return_value=False) as install:
                self.assertEqual(self.invoke_setup(arguments), 1)
                self.assertEqual(install.call_count, 1)

    def test_explicit_wrong_version_wheel_never_falls_back_to_pypi(self):
        with patch.object(setup, "pip_install") as install, contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(setup.install_carla("0.9.16", Path("carla-0.9.15-cp312-cp312-win_amd64.whl")))
            install.assert_not_called()

    def test_connected_install_failure_returns_nonzero(self):
        with patch.object(setup, "check_python_version"), \
                patch.object(setup, "remove_conflicting_opencv", return_value=True), \
                patch.object(setup, "detect_cuda_tag"), \
                patch.object(setup, "install_torch", return_value=False), \
                patch.object(setup, "install_requirements") as requirements:
            self.assertEqual(self.invoke_setup([]), 1)
            requirements.assert_not_called()


if __name__ == "__main__":
    unittest.main()
