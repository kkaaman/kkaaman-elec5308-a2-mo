"""Build an isolated CARLA 0.9.16 runtime with corrected RGB readback threading.

This utility supports one verified Windows build only. It never overwrites
the official executable or an unrecognized existing destination. A source-built
application-local D3D11 shim protects the immediate context shared by workers.
"""

import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
from pathlib import Path

ORIGINAL_SHA256 = "ba05ec36f7f0b548b266865ea2787a1dc5a028e298ebefcf0c49196ac629c4d3"
PATCHED_SHA256 = "9deba9928ffe56ee8c27cc89d88bcdd4e49204b771f86da2e438c613f4205245"
INSTRUCTION_RVA = 0x8945B8
BEFORE = bytes.fromhex("b9 00 02 00 00 e8 2e c1 7c 00")
AFTER = bytes.fromhex("b9 ff 02 00 00 e8 2e c1 7c 00")


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _patched_image(data):
    if _digest(data) != ORIGINAL_SHA256:
        raise ValueError("Unsupported CARLA executable SHA256; no files were changed")
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    if data[pe:pe + 4] != b"PE\0\0":
        raise ValueError("Invalid PE signature")
    machine, count = struct.unpack_from("<HH", data, pe + 4)
    if machine != 0x8664:
        raise ValueError("Expected a Windows x64 executable")
    optional_size = struct.unpack_from("<H", data, pe + 20)[0]
    offset = None
    for index in range(count):
        section = pe + 24 + optional_size + 40 * index
        name, _, address, size, pointer = struct.unpack_from("<8sIIII", data, section)
        if address <= INSTRUCTION_RVA < address + size:
            if name.rstrip(b"\0") != b".text":
                raise ValueError("Patch target is outside the code section")
            offset = pointer + INSTRUCTION_RVA - address
            break
    if offset is None or data[offset:offset + len(BEFORE)] != BEFORE:
        raise ValueError("Build-specific instruction guard failed")
    patched = data[:offset] + AFTER + data[offset + len(AFTER):]
    if _digest(patched) != PATCHED_SHA256:
        raise ValueError("Patched executable SHA256 verification failed")
    return patched, offset


def create_fixed_copy(carla_root, output=None):
    """Verify the official image and create, or verify, its separate fixed copy."""
    source = Path(carla_root).resolve() / "CarlaUE4/Binaries/Win64/CarlaUE4-Win64-Shipping.exe"
    destination = Path(output).resolve() if output else source.with_name("CarlaUE4-Win64-RgbFix.exe")
    if source == destination or (destination.exists() and source.samefile(destination)):
        raise ValueError("The destination must not replace the original executable")
    data = source.read_bytes()
    patched, offset = _patched_image(data)
    if destination.exists():
        if _digest(destination.read_bytes()) != PATCHED_SHA256:
            raise ValueError("Refusing to overwrite an unrecognized existing destination")
        status = "ALREADY_PRESENT"
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as file:
            file.write(patched)
        status = "CREATED"
    if _digest(destination.read_bytes()) != PATCHED_SHA256:
        raise ValueError("Written executable SHA256 verification failed")
    if _digest(source.read_bytes()) != ORIGINAL_SHA256:
        raise ValueError("The original executable changed during verification")
    return {
        "status": status,
        "source": str(source), "destination": str(destination),
        "original_sha256": ORIGINAL_SHA256, "patched_sha256": PATCHED_SHA256,
        "instruction_rva": hex(INSTRUCTION_RVA), "file_offset": hex(offset),
        "before": BEFORE.hex(), "after": AFTER.hex(), "changed_bytes": 1,
        "source_equivalent": "AsyncTask(ENamedThreads::AnyNormalThreadHiPriTask, ...)",
        "scope": "CARLA server readback task routing; no route/controller/configuration changes",
    }


def _junction(destination, source):
    def literal(path):
        return "'" + str(path).replace("'", "''") + "'"

    # Literal PowerShell paths prevent command substitution in user paths.
    command = (f"New-Item -ItemType Junction -Path {literal(destination)} "
               f"-Target {literal(source)} | Out-Null")
    subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
        check=True, capture_output=True, text=True,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )


def create_guarded_runtime(carla_root, runtime_root, compiler):
    """Create an isolated runtime; share assets through verified directory junctions."""
    carla_root, runtime_root = Path(carla_root).resolve(), Path(runtime_root).resolve()
    if runtime_root.is_relative_to(carla_root) or carla_root.is_relative_to(runtime_root):
        raise ValueError("The runtime must be outside the original CARLA installation")
    source = carla_root / "CarlaUE4/Binaries/Win64/CarlaUE4-Win64-Shipping.exe"
    if _digest(source.read_bytes()) != ORIGINAL_SHA256:
        raise ValueError("Unsupported CARLA executable SHA256; no files were changed")
    tools = Path(__file__).resolve().parents[1] / "tools/carla_rgb_readback"
    source_hashes = {name: _digest((tools / name).read_bytes())
                     for name in ("d3d11_guard.cpp", "d3d11_guard.def")}
    binary = runtime_root / "CarlaUE4/Binaries/Win64"
    manifest_path = runtime_root / "rgb-repair-runtime.json"
    if runtime_root.exists():
        if not manifest_path.is_file():
            raise ValueError("Refusing to modify an existing unrecognized runtime directory")
        report = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (not isinstance(report, dict)
                or report.get("kind") != "CARLA_RGB_READBACK_RUNTIME_V1"
                or report.get("guard_source_sha256") != source_hashes
                or report.get("carla_root") != str(carla_root)
                or report.get("runtime_root") != str(runtime_root)
                or report.get("executable") != str(binary / "CarlaUE4-Win64-Shipping.exe")
                or _digest((binary / "CarlaUE4-Win64-Shipping.exe").read_bytes()) != PATCHED_SHA256
                or _digest((binary / "d3d11.dll").read_bytes()) != report.get("guard_sha256")):
            raise ValueError("Existing runtime identity verification failed")
        for relative in ("Engine", "CarlaUE4/Content", "CarlaUE4/Config", "CarlaUE4/Plugins"):
            if (runtime_root / relative).resolve() != (carla_root / relative).resolve():
                raise ValueError("Existing runtime asset junction verification failed")
        report["status"] = "ALREADY_PRESENT"
        return report
    if os.name != "nt":
        raise ValueError("The guarded runtime builder requires Windows")
    if len(str(binary / "carla-d3d11-guard.jsonl")) >= 250:
        raise ValueError("Choose a shorter runtime path for the native diagnostic log")
    compiler = Path(compiler).resolve()
    if not compiler.is_file():
        raise ValueError("An existing MinGW-w64 g++ compiler is required")
    shared = {"Engine": carla_root / "Engine"}
    shared.update({f"CarlaUE4/{name}": carla_root / "CarlaUE4" / name
                   for name in ("Content", "Config", "Plugins")})
    if any(not path.is_dir() for path in shared.values()):
        raise ValueError("The CARLA installation is missing required asset directories")
    for name in ("CarlaUE4.uproject", "AssetRegistry.bin"):
        if not (carla_root / "CarlaUE4" / name).is_file():
            raise ValueError(f"The CARLA installation is missing {name}")
    report = create_fixed_copy(carla_root, binary / "CarlaUE4-Win64-Shipping.exe")
    for relative, source in shared.items():
        _junction(runtime_root / relative, source)
    for source in (carla_root / "CarlaUE4").iterdir():
        if source.is_file():
            shutil.copy2(source, runtime_root / "CarlaUE4" / source.name)
    for source in (carla_root / "CarlaUE4/Binaries/Win64").glob("*.dll"):
        if source.name.lower() != "d3d11.dll":
            shutil.copy2(source, binary / source.name)
    command = [str(compiler), "-shared", "-O2", "-s", "-static-libgcc", "-static-libstdc++",
               str(tools / "d3d11_guard.cpp"), str(tools / "d3d11_guard.def"),
               "-o", str(binary / "d3d11.dll")]
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=120,
                   creationflags=subprocess.CREATE_NO_WINDOW)
    guard = (binary / "d3d11.dll").read_bytes()
    pe = struct.unpack_from("<I", guard, 0x3C)[0]
    if guard[pe:pe + 4] != b"PE\0\0" or struct.unpack_from("<H", guard, pe + 4)[0] != 0x8664:
        raise ValueError("The compiler did not produce a Windows x64 DLL")
    report.update(
        kind="CARLA_RGB_READBACK_RUNTIME_V1", status="CREATED",
        carla_root=str(carla_root), runtime_root=str(runtime_root),
        executable=str(binary / "CarlaUE4-Win64-Shipping.exe"),
        guard_sha256=_digest(guard), guard_source_sha256=source_hashes,
        compiler=str(compiler), build_command=command,
        shared_assets={relative: str(source) for relative, source in shared.items()},
        launch_arguments="-dx11 -windowed -ResX=640 -ResY=360 -quality-level=Low -nosound",
        scope="Isolated server task-routing and D3D11 context protection; original executable unchanged",
    )
    manifest_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--carla-root", required=True, type=Path)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--cxx", required=True, type=Path,
                        help="Path to an existing Windows x64 MinGW-w64 g++ compiler")
    args = parser.parse_args(argv)
    try:
        report = create_guarded_runtime(args.carla_root, args.runtime, args.cxx)
    except (OSError, ValueError, struct.error, subprocess.SubprocessError) as error:
        print(json.dumps({"status": "FAIL", "error": str(error)}, indent=2))
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
