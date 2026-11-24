"""Pool variants100k_base_fix; paired comparisons with base and cd_call2 on the same deals."""
import collections, glob, json, math
import numpy as np

D = "checkpoints/qmmd_hunl_560k/local/"

def load(prefix):
  out, bad = {}, 0
  for p in sorted(glob.glob(D + prefix + "_s*.jsonl")):
    seed = int(p.rsplit("_s", 1)[1].split(".")[0])
    for line in open(p):
      try:
        r = json.loads(line)
      except json.JSONDecodeError:
        bad += 1
        continue
      out[(seed, r["hand"])] = r
  return out, bad

def ci(x):
  x = np.asarray(x, float)
  return x.mean(), 1.96 * x.std(ddof=1) / math.sqrt(len(x)), x.std(ddof=1)

def report(name, recs):
  w = {k: r["winnings"] * 10.0 for k, r in recs.items()}
  m, h, s = ci(list(w.values()))
  pairs = collections.defaultdict(list)
  for k, r in recs.items():
    pairs[(k[0], r["pair"])].append(w[k])
  full = [v for v in pairs.values() if len(v) == 2]
  pm, ph, _ = ci([np.mean(v) for v in full])
  a = np.array(full)
  sb = [w[k] for k, r in recs.items() if r["seat"] == 0]
  bb = [w[k] for k, r in recs.items() if r["seat"] == 1]
  lost = sorted(k for k, r in recs.items() if r.get("tree_lost") or r["resolving"].get("lost"))
  print(f"{name}: n={len(w)}  {m:+.0f} ± {h:.0f} (std {s / 1000:.2f} BB/h) | dup {len(full)} pairs "
        f"{pm:+.0f} ± {ph:.0f} corr {np.corrcoef(a[:, 0], a[:, 1])[0, 1]:+.2f} | "
        f"SB {ci(sb)[0]:+.0f} ± {ci(sb)[1]:.0f}  BB {ci(bb)[0]:+.0f} ± {ci(bb)[1]:.0f} | lost {lost}")
  return w

fix, bad = load("variants100k_base_fix")
print(f"base_fix: {bad} unparsable lines")
wf = report("base_fix", fix)
wb = report("base (old)", load("variants100k_base")[0])
wc = report("cd_call2", load("variants100k_cd_call2")[0])
for name, other in (("base_fix - cd_call2", wc), ("base_fix - base", wb)):
  ks = sorted(set(wf) & set(other))
  d = [wf[k] - other[k] for k in ks]
  m, h, s = ci(d)
  corr = np.corrcoef([wf[k] for k in ks], [other[k] for k in ks])[0, 1]
  print(f"{name}, paired by deal: n={len(ks)} {m:+.0f} ± {h:.0f}  sd {s / 1000:.2f} BB  corr {corr:+.2f}")
# The same comparison restricted to deals all three runs played.
ks = sorted(set(wf) & set(wb) & set(wc))
for name, w in (("base_fix", wf), ("base", wb), ("cd_call2", wc)):
  m, h, _ = ci([w[k] for k in ks])
  print(f"  on the {len(ks)} common deals: {name} {m:+.0f} ± {h:.0f}")
