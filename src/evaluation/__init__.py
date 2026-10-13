"""Run logging, metrics and evaluation-only ground truth (workstreams B and F).

Writes ``runs/<scenario>/<run-id>/`` (see ``autonomy.paths.run_dir``) and
computes the acceptance metrics. Ground truth enters here and never feeds the
control decision (BL-F-05 Saviour, BL-B-07 Mohammad).
"""
