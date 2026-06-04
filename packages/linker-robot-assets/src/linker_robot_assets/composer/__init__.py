"""Author-time workstation composer.

Reads component sources (arms, hands, bases, sensors) plus a recipe and
produces a monolithic workstation URDF and MJCF. Output is deterministic so
CI can drift-check committed artifacts against a fresh re-run.
"""

COMPOSER_VERSION = "0.1.0"
