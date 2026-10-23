"""Command-line entry point for the live mapless stack (G2/G3) on the drive harness."""

from carla_adapter.mapless_drive import main

if __name__ == "__main__":
    main(stack=True)
