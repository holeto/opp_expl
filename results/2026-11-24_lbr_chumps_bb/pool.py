"""Pool the chump logs of job 12088968 per cell (duplicate pairs, BB/hand ±95%).

  python pool.py <directory of the <chump>_<step>_min_<bets>_c<cs>_s<seed>.jsonl logs>
"""
import collections, glob, json, math, os, sys
import numpy as np

D = os.path.join(sys.argv[1] if len(sys.argv) > 1 else ".", "")
# The paper's Table 2 (BB/hand); uniform is compared with its random chump.
PAPER = {("call_raise", "fc", 0): "7.1 ± 0.5", ("call_raise", "fcpa", 0): "23.1 ± 0.5",
         ("call_raise", "fc", 2): "16.2 ± 0.3", ("call_raise", "fcpa", 2): "24.4 ± 0.6",
         ("random", "fc", 0): "15.7 ± 0.4", ("random", "fcpa", 0): "39.1 ± 0.6",
         ("random", "fc", 2): "2.2 ± 0.7", ("random", "fcpa", 2): "80.7 ± 0.7"}
for chump, step in [("call_raise", "bb"), ("random", "bb"), ("uniform", "chip"), ("uniform", "bb")]:
  for bets, cs in [("fc", 0), ("fcpa", 0), ("fc", 2), ("fcpa", 2)]:
    files = sorted(glob.glob(f"{D}{chump}_{step}_min_{bets}_c{cs}_s*.jsonl"))
    pairs, n = collections.defaultdict(list), 0
    for p in files:
      for line in open(p):
        r = json.loads(line)
        n += 1
        pairs[(p, r["pair"])].append(r["winnings"] / 100)
    pm = np.array([np.mean(v) for v in pairs.values() if len(v) == 2])
    if len(pm) < 2:
      print(f"{chump:10s} {step:4s} {bets:4s} rounds {'1-4' if cs == 0 else '3-4'}: no data")
      continue
    ci = 1.96 * pm.std(ddof=1) / math.sqrt(len(pm))
    # Per-hand stds: over all hands, and the duplicate one (pair averages × √2).
    hands = np.concatenate([v for v in pairs.values() if len(v) == 2])
    std, dup_std = hands.std(ddof=1), pm.std(ddof=1) * math.sqrt(2)
    paper = PAPER.get(("random" if chump == "uniform" else chump, bets, cs), "")
    print(f"{chump:10s} {step:4s} {bets:4s} rounds {'1-4' if cs == 0 else '3-4'}: files "
          f"{len(files):2d} hands {n:6d}  {pm.mean():+6.2f} ± {ci:4.2f} BB/hand  std {std:6.2f} dup {dup_std:6.2f}  paper {paper}")
