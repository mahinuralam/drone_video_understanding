"""Pre-extract 32 uniformly sampled frames per clip for 3 frame-offset grids (0, 1/3, 2/3 of the
inter-frame spacing) as JPEG, plus a per-video JSON with true fps, total frames, sampled indices and
pts times. 640x360 clips -> 640x352, 640x640 clips -> 480x480 (multiples of 32 = patch 16 x merge 2).
Output: frames/g{0,1,2}/<video_id>/NN.jpg and frames/meta/<video_id>.json
"""
import av, json, os, sys, time, argparse
import numpy as np
from PIL import Image
from multiprocessing import Pool
ap = argparse.ArgumentParser()
ap.add_argument("--nframes", type=int, default=32)
ap.add_argument("--min_sec", type=float, default=0.0, help="only cache clips longer than this (seconds)")
ap.add_argument("--out", default="frames/", help="output dir relative to the project root")
ap.add_argument("--hires", action="store_true", help="640x360 -> 1280x704 (2x upscale), 640x640 -> native 640x640")
ap.add_argument("--workers", type=int, default=16)
args = ap.parse_args()
R = os.path.join(os.environ.get("REVA_ROOT", os.path.expanduser("~/projects/reva")), "data/ReVA/")
OUT = os.path.join(os.environ.get("REVA_ROOT", os.path.expanduser("~/projects/reva")) + "/", args.out.rstrip("/") + "/")
NF = args.nframes; GRIDS = 3; Q = 92

def vid_id(vp): return vp.replace("/", "__").rsplit(".", 1)[0]

def target_size(w, h):
    if args.hires and (w, h) == (640, 360): return (1280, 704)
    if args.hires and (w, h) == (640, 640): return (640, 640)
    if (w, h) == (640, 360): return (640, 352)
    if (w, h) == (640, 640): return (480, 480)
    s = (230400 / (w * h)) ** 0.5
    return (max(32, round(w * s / 32) * 32), max(32, round(h * s / 32) * 32))

def grids(n):
    base = np.linspace(0, n - 1, NF); sp = (n - 1) / (NF - 1)
    return [np.clip(np.round(base + g * sp / GRIDS), 0, n - 1).astype(int).tolist() for g in range(GRIDS)]

def process(vp):
    vid = vid_id(vp); meta_path = OUT + "meta/" + vid + ".json"
    if os.path.exists(meta_path): return vid, "skip"
    try:
        c = av.open(R + vp); s = c.streams.video[0]; fps = float(s.average_rate)
        w, h = s.codec_context.width, s.codec_context.height
        times = [f.time for f in c.decode(s)]; c.close()
        n = len(times)
        if args.min_sec and n / fps <= args.min_sec: return vid, "short"
        gs = grids(n); need = {}
        for g, idx in enumerate(gs):
            os.makedirs(OUT + f"g{g}/" + vid, exist_ok=True)
            for j, i in enumerate(idx): need.setdefault(i, []).append((g, j))
        size = target_size(w, h)
        c = av.open(R + vp); s = c.streams.video[0]
        for i, f in enumerate(c.decode(s)):
            if i in need:
                img = f.to_image().resize(size, Image.BICUBIC)
                for g, j in need[i]: img.save(OUT + f"g{g}/{vid}/{j:02d}.jpg", quality=Q)
        c.close()
        meta = {"video_path": vp, "width": w, "height": h, "size": list(size), "fps": fps, "total_num_frames": n,
                "duration": n / fps, "stream_frames": s.frames, "grids": gs,
                "pts_times": [[times[i] for i in idx] for idx in gs]}
        json.dump(meta, open(meta_path, "w"))
        return vid, "ok"
    except Exception as e:
        return vid, "ERR " + repr(e)

if __name__ == "__main__":
    vids = sorted(set(q["video_path"] for sp in ["train", "val", "test"] for q in json.load(open(R + sp + ".json"))["QA"]))
    os.makedirs(OUT + "meta", exist_ok=True)
    t0 = time.time(); res = []
    with Pool(args.workers) as p:
        for k, (vid, st) in enumerate(p.imap_unordered(process, vids), 1):
            res.append((vid, st))
            if st.startswith("ERR"): print(vid, st, flush=True)
            if k % 100 == 0: print(f"{k}/{len(vids)} {time.time()-t0:.0f}s", flush=True)
    import collections; print(collections.Counter(s.split()[0] for _, s in res), f"{time.time()-t0:.0f}s")
