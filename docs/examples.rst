Worked Examples
===============

Traffic perception
------------------

The traffic perception module can filter traffic-specific detections and
optionally estimate distance for objects with a known real-world width.

Example:

.. code-block:: python

   from traffic_perception import TrafficPerception

   detector = TrafficPerception(
       backend=my_detection_backend,
       confidence_threshold=0.5,
   )

   detections = detector.detect(image)

   for detection in detections:
       print(
           detection.label,
           detection.confidence,
           detection.distance_m,
       )

CARLA demonstration
-------------------

For a complete simulator example, see :doc:`carla_demo`.
