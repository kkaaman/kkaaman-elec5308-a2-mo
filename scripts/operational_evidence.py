"""Read checksummed operational evidence without extracting archived paths."""

import hashlib
import json
import zipfile


def read_archived_evidence(folder, member="capture.json"):
    archive = folder / "raw-evidence.zip"
    expected = (folder / "raw-evidence.sha256").read_text().split()[0]
    if hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
        raise ValueError("Operational evidence archive hash mismatch")
    manifest = json.loads((folder / "raw-evidence.manifest.json").read_text())
    with zipfile.ZipFile(archive) as source:
        if set(source.namelist()) != set(manifest):
            raise ValueError("Operational evidence archive members mismatch")
        payload = None
        for name, record in manifest.items():
            data = source.read(name)
            if len(data) != record["bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
                raise ValueError("Operational evidence member hash mismatch")
            if name == member:
                payload = data
    if payload is None:
        raise ValueError("Missing operational evidence member")
    return payload
