"""Pool the rounds 1-4 chump logs per cell (duplicate pairs, BB/hand ±95%).

  python pool.py <directory of the <chump>_<bets>_c0_s<seed>.jsonl logs>
"""
import glob, json, collections, math, os, sys
import numpy as np
D = os.path.join(sys.argv[1] if len(sys.argv) > 1 else ".", "")
for cell in ["call_raise_fc", "call_raise_fcpa", "random_fc", "random_fcpa"]:
  files = sorted(glob.glob(D + cell + "_c0_s*.jsonl"))
  pairs = collections.defaultdict(list); n = 0
  for p in files:
    for line in open(p):
      r = json.loads(line); n += 1
      pairs[(p, r["pair"])].append(r["winnings"] / 100)
  pm = np.array([np.mean(v) for v in pairs.values() if len(v) == 2])
  ci = 1.96 * pm.std(ddof=1) / math.sqrt(len(pm))
  print(f"{cell:16s} files {len(files)} hands {n}  {pm.mean():+.2f} +- {ci:.2f} BB/hand ({len(pm)} pairs)")
