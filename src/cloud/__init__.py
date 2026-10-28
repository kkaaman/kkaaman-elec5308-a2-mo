"""Optional inference service, separate from the local baseline.

``cloud.server`` owns the bounded authenticated FastAPI service (issue #56).
The existing client and arbitration live in ``perception.remote_detector`` and
``perception.hybrid_detector`` (issue #57). Baseline use enables none of them.
"""
