"""AIVAT for the re-solver's slumbot.com logs; see ``oppex.policy_gradient.slumbot.offline_aivat``.

  uv run python scripts/policy_gradient/aivat_resolve_slumbot.py RUN_DIR LOG.jsonl --out A.jsonl
  uv run python scripts/policy_gradient/aivat_resolve_slumbot.py RUN_DIR --report A.jsonl
"""

from oppex.policy_gradient.slumbot.offline_aivat import main

if __name__ == "__main__":
  main()
