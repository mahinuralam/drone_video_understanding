"""Whole-recipe ensemble rule: for test questions whose exact question (same video, normalized text) exists with its answer in the
facts split shown to the Format B model, use the Format B member's averaged probabilities instead of the ensemble's.
Usage: dup_override.py --ens results/ens_fb_r1_3g_test.jsonl --fb "results/run5_9b_formatB_e2_test_g*_s0.jsonl" --facts splits/train_final.json --out results/ens_fb_r1_3g_dup_test.jsonl"""
import argparse, json, re, glob, collections, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import reva_common as C
ap = argparse.ArgumentParser(); ap.add_argument("--ens", required=True); ap.add_argument("--fb", required=True); ap.add_argument("--facts", default="splits/train_final.json")
ap.add_argument("--test", default="data/ReVA/test.json"); ap.add_argument("--out", required=True); args = ap.parse_args(); os.chdir(C.ROOT)
def norm(s): return re.sub(r"\s+", " ", s.strip().lower())
facts = collections.defaultdict(set)
for q in json.load(open(args.facts))["QA"]: facts[q["video_path"]].add(norm(q["question"]))
dup = {q["qa_id"] for q in json.load(open(args.test))["QA"] if norm(q["question"]) in facts.get(q["video_path"], set())}
acc = collections.defaultdict(lambda: [0.0] * 4); n = collections.Counter()
files = sorted(glob.glob(args.fb)); assert files, "no FB dumps"
for f in files:
    for l in open(f):
        r = json.loads(l); n[r["qa_id"]] += 1
        for i in range(4): acc[r["qa_id"]][i] += r["probs"][i]
fb = {q: [x / n[q] for x in p] for q, p in acc.items()}
rows = [json.loads(l) for l in open(args.ens)]; changed = flipped = 0
with open(args.out, "w") as fo:
    for r in rows:
        if r["qa_id"] in dup and r["qa_id"] in fb:
            newp = fb[r["qa_id"]]; newpred = C.LETTERS[max(range(4), key=lambda i: newp[i])]
            flipped += newpred != r["pred"]; r["probs"] = newp; r["pred"] = newpred; changed += 1
        fo.write(json.dumps(r) + "\n")
print(f"duplicated test questions: {len(dup)}; overridden with Format B probabilities: {changed}; predictions flipped: {flipped}; wrote {args.out}")
