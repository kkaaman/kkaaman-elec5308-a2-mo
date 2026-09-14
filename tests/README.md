# Tests

This directory contains unit tests for the project's simulator-independent library components.

## Running the tests

Run the complete test suite from the repository root:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## Navigation-control tests

The path-following, steering and speed-control tests are in
`tests/navigation_control/`. Run them from the repository root with:

```bash
python -m pytest tests/navigation_control -q
```

This test group configures its own `src` import path while the team-level
package configuration is being finalised.
