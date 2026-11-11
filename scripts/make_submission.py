"""Average per-option probabilities over eval dumps (grids x shifts x models; optional per-dump weight as
path:weight, optional per-task weights JSON) and write predictions.json inside a zip.
Usage: make_submission.py --dumps results/a.jsonl results/b.jsonl:0.5 --out submissions/run1_e0.zip [--test data/ReVA/test.json]
"""
import argparse, json, os, sys, zipfile, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reva_common as C

ap = argparse.ArgumentParser()
ap.add_argument("--dumps", nargs="+", required=True)
ap.add_argument("--test", default="data/ReVA/test.json")
ap.add_argument("--out", required=True)
ap.add_argument("--task_weights", default="", help="JSON {dump_path: {task: weight}} for category-level ensemble weights")
ap.add_argument("--allow_missing", action="store_true", help="fill missing test qa_ids with A instead of failing")
ap.add_argument("--format", default="testjson", choices=["testjson", "map"],
                help="testjson = the test.json structure with correct_answer filled (Codabench Data page: 'fill this field'); map = {qa_id: letter}")
args = ap.parse_args(); os.chdir(C.ROOT)
tw = json.load(open(args.task_weights)) if args.task_weights else {}
acc = collections.defaultdict(lambda: [0.0] * 4); wsum = collections.defaultdict(float); meta = {}
for d in args.dumps:
    path, w = (d.rsplit(":", 1) + ["1"])[:2] if d.count(":") and d.rsplit(":", 1)[1].replace(".", "").isdigit() else (d, "1")
    w = float(w)
    for line in open(path):
        r = json.loads(line); qid = r["qa_id"]; ww = w * tw.get(path, {}).get(r["task"], 1.0)
        if len(args.dumps) == 1 and "pred" in r:  # a single (merged) dump must be self-consistent
            assert r["pred"] == C.LETTERS[max(range(4), key=lambda i: r["probs"][i])], f"{qid}: pred {r['pred']} != argmax(probs)"
        for i in range(4): acc[qid][i] += ww * r["probs"][i]
        wsum[qid] += ww; meta[qid] = (r["task"], r.get("gt", ""))
pred = {}
for qid, p in acc.items():
    p = [x / wsum[qid] for x in p]; pred[qid] = C.LETTERS[max(range(4), key=lambda i: p[i])]
test_ids = [q["qa_id"] for q in json.load(open(args.test))["QA"]]
missing = [q for q in test_ids if q not in pred]
if missing:
    print(f"{'WARNING' if args.allow_missing else 'ERROR'}: {len(missing)} test qa_ids missing from dumps (e.g. {missing[:3]})")
    if not args.allow_missing: sys.exit(1)
extra = set(pred) - set(test_ids)
assert not extra, f"{len(extra)} unknown qa_ids in dumps (e.g. {sorted(extra)[:3]})"
out = {q: pred.get(q, "A") for q in test_ids}
rows = [{"task": meta[q][0], "pred": pred[q], "gt": meta[q][1]} for q in pred if meta[q][1]]
if rows: print("ensemble accuracy on labelled rows:\n" + C.fmt_acc(C.per_task_accuracy(rows)))
print("letter distribution:", dict(sorted(collections.Counter(out.values()).items())), "n =", len(out))
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
if args.format == "testjson":
    test = json.load(open(args.test))
    for q in test["QA"]: q["correct_answer"] = out[q["qa_id"]]
    payload = json.dumps(test, indent=2, ensure_ascii=False)
else:
    payload = json.dumps(out, indent=0)
with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z: z.writestr("predictions.json", payload)
open(args.out.replace(".zip", ".predictions.json"), "w").write(payload)
print("wrote", args.out)
