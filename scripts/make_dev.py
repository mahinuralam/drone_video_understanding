"""Carve a DEV set from train, stratified to the test task mix (= 2x val task counts).

Rules:
- key = (video_path, normalized question). Keys with >1 train row (54 keys, conflicting option sets)
  are never placed in dev.
- Train keys that also appear in test (24) are never placed in dev (they stay in training).
- dev target per task = round(test_count * 1500 / 4000).
- train_run1 = train + val, minus dev keys, deduplicated by (key, option-text set, answer text), i.e. letter-permuted copies count once.
- train_final = train + val, exact-row deduplicated (dev folded back in).
- val_nontrain = the val rows whose key is not in train (secondary check only).
"""
import json, re, random, collections, os, sys
random.seed(0)
R = os.path.join(os.environ.get("REVA_ROOT", os.path.expanduser("~/projects/reva")), "data/ReVA/")
O = os.path.join(os.environ.get("REVA_ROOT", os.path.expanduser("~/projects/reva")), "splits/")
tr = json.load(open(R + "train.json"))["QA"]; va = json.load(open(R + "val.json"))["QA"]; te = json.load(open(R + "test.json"))["QA"]

def norm(s): return re.sub(r"\s+", " ", s.strip().lower())
def key(q): return (q["video_path"], norm(q["question"]))
def rowkey(q): return key(q) + (tuple(sorted(q["options"].values())), q["options"][q["correct_answer"]])  # permutation-invariant

test_counts = collections.Counter(q["task"] for q in te)
target = {t: round(c * 1500 / 4000) for t, c in test_counts.items()}
groups = collections.defaultdict(list)
for q in tr: groups[key(q)].append(q)
test_keys = set(map(key, te)); train_keys = set(groups)
cands = collections.defaultdict(list)
for k, g in groups.items():
    if len(g) == 1 and k not in test_keys: cands[g[0]["task"]].append(g[0])
dev = []
for t in sorted(target):
    c = cands[t]; random.shuffle(c)
    assert len(c) >= target[t], (t, len(c), target[t])
    dev += c[:target[t]]
dev.sort(key=lambda q: q["qa_id"])
dev_keys = set(map(key, dev))

def dedup(rows, exclude_keys):
    seen, out = set(), []
    for q in rows:
        if key(q) in exclude_keys: continue
        rk = rowkey(q)
        if rk in seen: continue
        seen.add(rk); out.append(q)
    return out
train_run1 = dedup(tr + va, dev_keys)
train_final = dedup(tr + va, set())
val_nontrain = [q for q in va if key(q) not in train_keys]
stats = {
    "dev": {"n": len(dev), "videos": len(set(q["video_path"] for q in dev)), "tasks": dict(sorted(collections.Counter(q["task"] for q in dev).items())),
            "answers": dict(sorted(collections.Counter(q["correct_answer"] for q in dev).items())),
            "keys_also_in_val": sum(key(q) in set(map(key, va)) for q in dev)},
    "train_run1": {"n": len(train_run1), "videos": len(set(q["video_path"] for q in train_run1)), "tasks": dict(sorted(collections.Counter(q["task"] for q in train_run1).items()))},
    "train_final": {"n": len(train_final)},
    "val_nontrain": {"n": len(val_nontrain), "keys_in_test": sum(key(q) in test_keys for q in val_nontrain)},
    "target": target, "seed": 0,
}
os.makedirs(O, exist_ok=True)
for name, rows in [("dev", dev), ("train_run1", train_run1), ("train_final", train_final), ("val_nontrain", val_nontrain)]:
    json.dump({"QA": rows}, open(O + name + ".json", "w"))
json.dump(stats, open(O + "stats.json", "w"), indent=1)
print(json.dumps(stats, indent=1))
