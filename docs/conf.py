import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

project = "Autonomous Perception and Navigation Library"
author = "ELEC5308 Group 7"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "myst_parser",
]

autodoc_mock_imports = ["carla"]

templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "alabaster"

autodoc_member_order = "bysource"
