"""Safety and reproducibility of the narrowly supported executable-copy repair."""

import hashlib
import struct

import pytest

from scripts import patch_carla_rgb_readback as repair


@pytest.fixture
def supported_image(tmp_path, monkeypatch):
    # A compact PE fixture exercises guards without shipping a simulator binary.
    image = bytearray(0xA00)
    struct.pack_into("<I", image, 0x3C, 0x80)
    image[0x80:0x84] = b"PE\0\0"
    struct.pack_into("<HH", image, 0x84, 0x8664, 1)
    struct.pack_into("<H", image, 0x94, 0xF0)
    struct.pack_into("<8sIIII", image, 0x188, b".text", 0x800, 0x894000, 0x800, 0x200)
    offset = 0x7B8
    image[offset:offset + len(repair.BEFORE)] = repair.BEFORE
    patched = image[:offset] + repair.AFTER + image[offset + len(repair.AFTER):]
    monkeypatch.setattr(repair, "ORIGINAL_SHA256", hashlib.sha256(image).hexdigest())
    monkeypatch.setattr(repair, "PATCHED_SHA256", hashlib.sha256(patched).hexdigest())
    source = tmp_path / "CarlaUE4/Binaries/Win64/CarlaUE4-Win64-Shipping.exe"
    source.parent.mkdir(parents=True)
    source.write_bytes(image)
    return tmp_path, source, bytes(image), bytes(patched)


def test_separate_copy_changes_only_task_target_and_preserves_original(supported_image):
    root, source, original, expected = supported_image
    report = repair.create_fixed_copy(root)
    destination = source.with_name("CarlaUE4-Win64-RgbFix.exe")
    assert report["status"] == "CREATED"
    assert destination.read_bytes() == expected
    assert source.read_bytes() == original
    assert sum(a != b for a, b in zip(original, expected)) == 1


def test_unsupported_original_is_rejected_without_creating_output(supported_image):
    root, source, original, _ = supported_image
    source.write_bytes(original + b"different build")
    with pytest.raises(ValueError, match="Unsupported CARLA executable SHA256"):
        repair.create_fixed_copy(root)
    assert not source.with_name("CarlaUE4-Win64-RgbFix.exe").exists()
    assert source.read_bytes() == original + b"different build"


def test_instruction_guard_rejects_unexpected_code(supported_image, monkeypatch):
    root, source, original, _ = supported_image
    changed = bytearray(original)
    changed[0x7B8] ^= 1
    source.write_bytes(changed)
    monkeypatch.setattr(repair, "ORIGINAL_SHA256", hashlib.sha256(changed).hexdigest())
    with pytest.raises(ValueError, match="instruction guard"):
        repair.create_fixed_copy(root)
    assert not source.with_name("CarlaUE4-Win64-RgbFix.exe").exists()


def test_existing_unrecognized_copy_is_never_overwritten(supported_image):
    root, source, original, _ = supported_image
    destination = source.with_name("CarlaUE4-Win64-RgbFix.exe")
    destination.write_bytes(b"existing user file")
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        repair.create_fixed_copy(root)
    assert destination.read_bytes() == b"existing user file"
    assert source.read_bytes() == original


def test_verified_copy_is_idempotent(supported_image):
    root, source, original, expected = supported_image
    repair.create_fixed_copy(root)
    report = repair.create_fixed_copy(root)
    assert report["status"] == "ALREADY_PRESENT"
    assert source.read_bytes() == original
    assert source.with_name("CarlaUE4-Win64-RgbFix.exe").read_bytes() == expected


def test_original_cannot_be_used_as_destination(supported_image):
    root, source, original, _ = supported_image
    with pytest.raises(ValueError, match="must not replace"):
        repair.create_fixed_copy(root, source)
    assert source.read_bytes() == original


def test_guard_runtime_cannot_modify_the_original_installation(supported_image):
    root, source, original, _ = supported_image
    with pytest.raises(ValueError, match="outside the original"):
        repair.create_guarded_runtime(root, root / "runtime", root / "compiler.exe")
    assert source.read_bytes() == original
    assert not (root / "runtime").exists()


def test_guard_runtime_rejects_an_existing_unrecognized_directory(supported_image):
    root, _, _, _ = supported_image
    runtime = root.with_name(root.name + "-existing-runtime")
    runtime.mkdir()
    marker = runtime / "keep.txt"
    marker.write_text("existing user file")
    with pytest.raises(ValueError, match="existing unrecognized"):
        repair.create_guarded_runtime(root, runtime, root / "compiler.exe")
    assert marker.read_text() == "existing user file"
