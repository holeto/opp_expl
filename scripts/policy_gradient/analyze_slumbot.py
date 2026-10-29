"""Summarise ``play_slumbot.py`` logs; see ``oppex.policy_gradient.slumbot.analyze``.

  uv run python scripts/policy_gradient/analyze_slumbot.py checkpoints/qmmd_hunl_560k/slumbot/*.jsonl
"""

from oppex.policy_gradient.slumbot.analyze import main

if __name__ == "__main__":
  main()
