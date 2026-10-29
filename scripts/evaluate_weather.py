"""Score retained E1-03 car/person detections across the frozen weather matrix."""

import argparse
import json
from pathlib import Path

from evaluation.weather import evaluate_weather, markdown_table


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    args = parser.parse_args(argv)

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    report = evaluate_weather(
        json.loads(args.input.read_text(), object_pairs_hook=unique),
        json.loads(args.matrix.read_text(), object_pairs_hook=unique),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.table.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    with args.table.open("x") as stream:
        stream.write(f"# E1-03 weather perception table — {report['status']}\n\n")
        stream.write(markdown_table(report))
    print(
        f"Weather audit {report['status']}: {len(report['missing_cells'])} missing cells, "
        f"{len(report['missing_denominators'])} missing denominators"
    )
    return 0 if report["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
