"""Read-only client/server and optional GPU prerequisite check for DOC-03.

Does not spawn actors, tick, change world settings or download weights.
"""

import argparse
import json


def check_versions(client, expected):
    """Read actual client/server strings; reject even a mutually matching wrong version."""
    versions = {"client": client.get_client_version(), "server": client.get_server_version()}
    if any(value != expected for value in versions.values()):
        raise ValueError(f"Expected CARLA {expected}; observed {versions}")
    return versions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--expected-version", default="0.9.16")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    if not 0 < args.timeout <= 60:
        parser.error("--timeout must be greater than zero and at most 60 seconds")
    report = {"status": "FAIL"}
    try:
        import carla

        client = carla.Client(args.host, args.port)
        client.set_timeout(args.timeout)
        report.update(check_versions(client, args.expected_version))
        if args.require_cuda:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable to this torch build; use a validated GPU environment")
            report["gpu"] = torch.cuda.get_device_name(0)
            report["torch"] = torch.__version__
        report["status"] = "PASS"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["remedy"] = "Check the installed CARLA wheel, running server/version and host/port; see docs/reproduction.md"
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
