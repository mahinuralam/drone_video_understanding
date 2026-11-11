"""Extension of the duplicate rule: test questions whose wording is a near-duplicate (difflib ratio >= --thr) of a training
question on the same video use the Format B members average (they see that training QA as a fact). Dev (run1 vs run5, sim >= 0.8,
95 q): plain 77, avg 80, Format B 82."""
import argparse, json, re, glob, collections, difflib, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import reva_common as C
ap = argparse.ArgumentParser(); ap.add_argument("--ens", required=True); ap.add_argument("--fb", required=True)
ap.add_argument("--facts", default="splits/train_final.json"); ap.add_argument("--test", default="data/ReVA/test.json")
ap.add_argument("--thr", type=float, default=0.8); ap.add_argument("--out", required=True); a = ap.parse_args(); os.chdir(C.ROOT)
norm = lambda s: re.sub(r"\s+", " ", s.strip().lower())
byv = collections.defaultdict(list)
for q in json.load(open(a.facts))["QA"]: byv[q["video_path"]].append(norm(q["question"]))
near = {q["qa_id"] for q in json.load(open(a.test))["QA"]
        if max([difflib.SequenceMatcher(None, norm(q["question"]), x).ratio() for x in byv.get(q["video_path"], [])], default=0) >= a.thr}
acc = collections.defaultdict(lambda: [0.0]*4); n = collections.Counter()
for f in sorted(glob.glob(a.fb)):
    for l in open(f):
        r = json.loads(l); n[r["qa_id"]] += 1
        for i in range(4): acc[r["qa_id"]][i] += r["probs"][i]
flipped = 0
with open(a.out, "w") as fo:
    for l in open(a.ens):
        r = json.loads(l)
        if r["qa_id"] in near:
            p = [x / n[r["qa_id"]] for x in acc[r["qa_id"]]]; new = C.LETTERS[max(range(4), key=lambda i: p[i])]
            flipped += new != r["pred"]; r["probs"], r["pred"] = p, new
        fo.write(json.dumps(r) + "\n")
print(f"near-duplicate (>= {a.thr}) test questions: {len(near)}; predictions flipped: {flipped}; wrote {a.out}")
