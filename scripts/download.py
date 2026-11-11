"""Download the ReVA videos + QA JSONs (Hugging Face ReVA-Benchmark/ReVA) into $REVA_ROOT/data/ReVA and the base model
Qwen/Qwen3.5-9B (pinned revision) into the Hugging Face cache."""
import json, os, time
from huggingface_hub import snapshot_download, hf_hub_download
ROOT = os.environ.get("REVA_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
root = os.path.join(ROOT, "data/ReVA"); os.makedirs(root, exist_ok=True)
BASE, REV = "Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
t0 = time.time(); paths = set()
for f in ["train.json", "val.json", "test.json", "README.md"]:
    hf_hub_download("ReVA-Benchmark/ReVA", f, repo_type="dataset", local_dir=root)
for f in ["train.json", "val.json", "test.json"]:
    for q in json.load(open(os.path.join(root, f)))["QA"]: paths.add(q["video_path"])
print(f"{len(paths)} distinct videos referenced by the splits", flush=True)
snapshot_download("ReVA-Benchmark/ReVA", repo_type="dataset", local_dir=root, allow_patterns=sorted(paths), max_workers=8)
missing = [p for p in paths if not os.path.isfile(os.path.join(root, p))]
print(f"videos done in {time.time()-t0:.0f}s, missing={len(missing)} {missing[:5]}", flush=True)
p = snapshot_download(BASE, revision=REV, max_workers=8)
print(f"{BASE}@{REV[:8]} at {p}", flush=True)
