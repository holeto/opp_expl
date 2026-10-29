"""Play a trained QMMD HUNL checkpoint against Slumbot; see ``oppex.policy_gradient.slumbot.play``.

  uv run python scripts/policy_gradient/play_slumbot.py checkpoints/qmmd_hunl_560k --hands 10
"""

from oppex.policy_gradient.slumbot.play import main

if __name__ == "__main__":
  main()
