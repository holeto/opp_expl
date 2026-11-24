"""Pool lbr100k logs per variant; paired comparison with variants100k_cd_call2."""
import collections, glob, json, math, sys
import numpy as np

D = "checkpoints/qmmd_hunl_560k/local/"

def load(prefix):
  out = {}
  stats = collections.Counter()
  for p in sorted(glob.glob(D + prefix + "_s*.jsonl")):
    seed = int(p.rsplit("_s", 1)[1].split(".")[0])
    for line in open(p):
      r = json.loads(line)
      out[(seed, r["hand"])] = r
  return out

def ci(x):
  x = np.asarray(x, float)
  return x.mean(), 1.96 * x.std(ddof=1) / math.sqrt(len(x)), x.std(ddof=1)

def report(name, recs, scale):
  w = {k: r["winnings"] * scale for k, r in recs.items()}
  m, h, s = ci(list(w.values()))
  pairs = collections.defaultdict(list)
  for k, r in recs.items():
    pairs[(k[0], r["pair"])].append(w[k])
  pm = [np.mean(v) for v in pairs.values() if len(v) == 2]
  a = np.array([v for v in pairs.values() if len(v) == 2])
  pmm, ph, _ = ci(pm)
  corr = np.corrcoef(a[:, 0], a[:, 1])[0, 1]
  sb = [w[k] for k, r in recs.items() if r["seat"] == 0]
  bb = [w[k] for k, r in recs.items() if r["seat"] == 1]
  print(f"{name}: n={len(w)}  {m:+.0f} ± {h:.0f}  (std {s/1000:.2f} BB/h)  dup pairs {len(pm)}: "
        f"{pmm:+.0f} ± {ph:.0f} corr {corr:+.2f}  SB {ci(sb)[0]:+.0f} ± {ci(sb)[1]:.0f}  "
        f"BB {ci(bb)[0]:+.0f} ± {ci(bb)[1]:.0f}")
  c = collections.Counter()
  for r in recs.values():
    c["lost"] += bool(r.get("tree_lost")) or bool(r.get("resolving", {}).get("lost"))
    for st in r.get("resolving", {}).get("steps", []):
      l = st.get("lbr")
      if l:
        c["lbr_dec"] += 1
        c["ch_" + l["choice"]] += 1
        if abs(l["pot"] - l["tree_pot"]) > 0.5:
          c["pot_mismatch"] += 1
        c["st%d" % st["street"]] += 1
  print("   ", dict(sorted(c.items())))
  return w

paths = sys.argv[1:]
res = {}
for v in ["fcpa_c2", "bins_c2"]:
  res[v] = report(v, load("lbr100k_" + v), 10.0)  # chips, BB = 100 -> mbb
ac = load("lbr100k_fcpa_c0_always_call")
report("fcpa_c0 vs always_call (BB/h x1000 = mbb)", ac, 10.0)
cd = report("cd_call2", load("variants100k_cd_call2"), 10.0)
for v in res:
  ks = sorted(set(res[v]) & set(cd))
  d = [res[v][k] - cd[k] for k in ks]
  m, h, s = ci(d)
  corr = np.corrcoef([res[v][k] for k in ks], [cd[k] for k in ks])[0, 1]
  print(f"{v} - cd_call2 paired by deal: n={len(ks)} {m:+.0f} ± {h:.0f}  sd {s/1000:.2f} BB  corr {corr:+.2f}")
