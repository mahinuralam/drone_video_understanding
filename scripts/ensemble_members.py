"""Final ensemble: per-question option probabilities averaged over the 3 frame grids of each member, then over the members
with equal weight. Output feeds dup_override.py.
Usage: ensemble_members.py --members "results/final_9b_formatB_test_g*.jsonl" "results/final_9b_plain_test_g*.jsonl" ... --out results/ens_test.jsonl"""
import argparse, json, glob, collections, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import reva_common as C
ap = argparse.ArgumentParser(); ap.add_argument("--members", nargs="+", required=True); ap.add_argument("--out", required=True)
args = ap.parse_args(); os.chdir(C.ROOT)
def avg(pattern):
    acc = collections.defaultdict(lambda: [0.0] * 4); n = collections.Counter(); meta = {}
    files = sorted(glob.glob(pattern)); assert len(files) == 3, (pattern, files)
    for f in files:
        for l in open(f):
            r = json.loads(l); n[r["qa_id"]] += 1; meta[r["qa_id"]] = r["task"]
            for i in range(4): acc[r["qa_id"]][i] += r["probs"][i]
    return {q: [x / n[q] for x in p] for q, p in acc.items()}, meta
M = [avg(p) for p in args.members]; meta = M[0][1]
assert all(len(m) == 4000 for m, _ in M), [len(m) for m, _ in M]
with open(args.out, "w") as fo:
    for q in sorted(M[0][0]):
        p = [sum(m[q][i] for m, _ in M) / len(M) for i in range(4)]
        fo.write(json.dumps({"qa_id": q, "task": meta[q], "gt": "", "pred": C.LETTERS[max(range(4), key=lambda i: p[i])], "probs": p}) + "\n")
print(f"ensemble of {len(M)} members written to {args.out}")
