# CARLA Navigation Demo

This guide runs the group navigation controller through the CARLA adapter.

## What the demo does

The script:

1. Connects to a running CARLA server.
2. Spawns one vehicle.
3. Builds a lane-centre route from CARLA map waypoints.
4. Converts CARLA vehicle data into `VehicleState`.
5. Uses `PathFollower` to calculate steering, throttle and brake.
6. Applies the resulting command to the CARLA vehicle.

## Requirements

- Windows 64-bit computer with a suitable dedicated GPU for CARLA.
- Python 3.12.
- CARLA Python API version 0.9.16.
- CARLA simulator version 0.9.16.
- The repository dependencies installed with:

```powershell
python -m pip install -e ".[test,carla]"