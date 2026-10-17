"""Append-only run records with explicit incomplete status and provenance."""

import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path


def json_text(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_identity(repo):
    result = {"python": platform.python_version()}
    for name, command in (
        ("commit", ["git", "rev-parse", "HEAD"]),
        ("working_changes", ["git", "status", "--porcelain"]),
    ):
        try:
            process = subprocess.run(command, cwd=repo, capture_output=True, text=True, timeout=10)
            result[name] = process.stdout.strip() if process.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            result[name] = None
    # A commit alone cannot identify uncommitted code. Record actual source hash.
    digest = hashlib.sha256()
    for relative in sorted(
        [
            *Path(repo).glob("src/**/*.py"),
            *Path(repo).glob("scripts/**/*.py"),
            *Path(repo).glob("configs/**/*"),
            Path(repo) / "pyproject.toml",
        ]
    ):
        if relative.is_file() and "__pycache__" not in relative.parts:
            digest.update(str(relative.relative_to(repo)).replace("\\", "/").encode())
            digest.update(relative.read_bytes().replace(b"\r\n", b"\n"))
    result["source_sha256"] = digest.hexdigest()
    return result


class RunLogger:
    def __init__(self, folder, manifest):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=False)
        self.manifest = dict(
            manifest, status="INCOMPLETE", started_wall_ns=time.time_ns(), schema_version=1
        )
        self._save_manifest()
        self.samples = (self.folder / "samples.jsonl").open("x", encoding="utf-8")
        self.events = (self.folder / "events.jsonl").open("x", encoding="utf-8")
        self._last_frame, self._last_time = -1, -1.0
        self._closed = False

    def _save_manifest(self):
        temp = self.folder / "manifest.tmp"
        temp.write_text(json_text(self.manifest) + "\n", encoding="utf-8")
        temp.replace(self.folder / "manifest.json")

    def sample(self, record):
        if record["frame_id"] <= self._last_frame or record["sim_time_s"] <= self._last_time:
            raise ValueError("Log frames and simulation time must increase")
        text = json_text(record)
        self.samples.write(text + "\n")
        self.samples.flush()
        self._last_frame, self._last_time = record["frame_id"], record["sim_time_s"]

    def event(self, record):
        self.events.write(json_text(record) + "\n")
        self.events.flush()

    def close(self, *, completed, cleanup, error=None):
        if self._closed:
            return
        self.samples.close()
        self.events.close()
        self._closed = True
        self.manifest.update(
            status="COLLECTED" if completed and not error and cleanup.get("ok") else "INCOMPLETE",
            error=error,
            cleanup=cleanup,
            ended_wall_ns=time.time_ns(),
        )
        self.manifest["files_sha256"] = {
            name: file_sha256(self.folder / name) for name in ("samples.jsonl", "events.jsonl")
        }
        self._save_manifest()
