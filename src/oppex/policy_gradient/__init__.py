"""Policy-gradient baselines and the blueprint they provide for search.

``algorithms`` holds QMMD (Layer 5) and ``losses`` its per-sample losses.
``blueprint`` loads a trained checkpoint as a playing policy together with the
Pluribus-style biased continuations, ``leaf_values`` samples those
continuations into matrix-valued leaves for ``oppex.cfr``, and ``slumbot``
plays a checkpoint against Slumbot over HTTP.

Only the algorithms are re-exported here; the other modules pull in the config
builders and are imported explicitly.
"""

from .algorithms import QMMD, Algorithm, TrainingState

__all__ = ["QMMD", "Algorithm", "TrainingState"]
