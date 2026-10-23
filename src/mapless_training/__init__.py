"""Training-side code for the mapless stack: datasets from the capture runs and the
training loops. This package reads labels and truth; nothing in ``src/mapless`` may
import it (the firewall test checks), and the deployed model loader never opens a label
folder (plan v2.1, section 12)."""
