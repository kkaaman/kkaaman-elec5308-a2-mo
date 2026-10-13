"""Optional hybrid inference backend (extension E4). Off by default.

Planned modules:

* ``remote_detector.py``: nonblocking HTTP/JPEG detector client
  (E4-04 Saviour)
* ``hybrid_detector.py``: concurrent local fallback and source selection
  (E4-05 Saviour)
* ``server/``: inference server with isolated tracker sessions and a
  Dockerfile (E4-02, E4-03 Ricky)

The local baseline must run with this package unused and no network.
"""
