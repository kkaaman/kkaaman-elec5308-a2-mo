import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "Autonomous Perception and Navigation Library"
author = "ELEC5308 Group 7"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "myst_parser",
]

autodoc_mock_imports = ["carla"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "evidence/BL-C-04-06/ai-journal.md"]

html_theme = "alabaster"

autodoc_member_order = "bysource"

exclude_patterns = ["_build", "evidence/TEST-13/ai-journal.md"]
