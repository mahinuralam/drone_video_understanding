"""Shared prompt / data helpers for the ReVA entry. Same template at train and test."""
import json, os, random
import torch
from PIL import Image

ROOT = os.environ.get("REVA_ROOT", os.path.expanduser("~/projects/reva"))
FRAMES = os.environ.get("REVA_FRAMES", ROOT + "/frames")  # REVA_FRAMES: alternate 32-frame cache (e.g. frames_hr/)
NFRAMES = 32
SYSTEM = "You are watching a drone video. Answer the multiple-choice question with only the letter."
LETTERS = "ABCD"
VIDEO_PLACEHOLDER = "<|vision_start|><|video_pad|><|vision_end|>"
THINK = "<think>\n\n</think>\n\n"  # empty think block (Qwen3.5 non-thinking mode)
TASKS = ["Object and Land Cover Recognition", "Temporal Grounding", "Change Detection", "Perspective and Viewpoint",
         "Geometric Relation", "Trend and Pattern", "Structural Layout", "General Understanding",
         "Causation Reasoning", "Hypothetical Reasoning", "Consequence Reasoning"]

def vid_id(vp): return vp.replace("/", "__").rsplit(".", 1)[0]

def frames_dir(nframes):
    return FRAMES if nframes == NFRAMES else f"{FRAMES}{nframes}"  # frames/ (32) or frames48/ etc.

def load_video(vp, grid, long_frames=0, long_sec=8.0):
    """Cached frames (list of PIL) + the video_metadata dict the Qwen3-VL processor needs.
    long_frames > 0: clips longer than long_sec use the long_frames-frame cache (frames<long_frames>/)."""
    vid = vid_id(vp)
    d = FRAMES
    if long_frames:
        base = json.load(open(f"{FRAMES}/meta/{vid}.json"))
        if base["duration"] > long_sec and os.path.exists(f"{frames_dir(long_frames)}/meta/{vid}.json"): d = frames_dir(long_frames)
    meta = json.load(open(f"{d}/meta/{vid}.json"))
    frames = [Image.open(f"{d}/g{grid}/{vid}/{j:02d}.jpg").convert("RGB") for j in range(len(meta["grids"][grid]))]
    vmeta = {"fps": meta["fps"], "total_num_frames": meta["total_num_frames"],
             "frames_indices": meta["grids"][grid], "duration": meta["duration"]}
    return frames, vmeta

def cyclic_perm(shift):
    """perm[i] = original letter shown at new position i. shift=1 -> new A = old B."""
    return tuple(LETTERS[(i + shift) % 4] for i in range(4))

def permute_options(q, perm):
    opts = {LETTERS[i]: q["options"][perm[i]] for i in range(4)}
    ans = q.get("correct_answer") or None
    return opts, (LETTERS[perm.index(ans)] if ans else None)

def question_text(question, opts):
    return "Question: " + question.strip() + "\n" + "\n".join(f"{k}. {opts[k]}" for k in LETTERS) + "\nAnswer:"

def prefix_text(facts=""):
    """facts: optional 'known facts' block (same-video QAs) placed after the video, inside the cached prefix."""
    return f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{VIDEO_PLACEHOLDER}" + facts

def facts_block(qas, max_facts=40):
    """Format B context: other QAs about the same video as known facts (question, options, answer)."""
    if not qas: return ""
    lines = []
    for q in qas[:max_facts]:
        opts = "; ".join(f"{k}. {q['options'][k]}" for k in LETTERS)
        lines.append(f"Q: {q['question'].strip()} [{opts}] Answer: {q['correct_answer']}")
    return "Known facts about this video from other questions:\n" + "\n".join(lines) + "\n\n"

def suffix_text(qtext, think=False):
    return f"{qtext}<|im_end|>\n<|im_start|>assistant\n" + (THINK if think else "")

def build_text(qtexts, answers, think=False, facts=""):
    """Packed multi-turn ChatML text: system, user(video [+ facts] + Q1), assistant(A1), user(Q2), assistant(A2), ...
    If len(answers) == len(qtexts) - 1 the last assistant turn is left open (inference)."""
    text = f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
    for i, qt in enumerate(qtexts):
        text += "<|im_start|>user\n" + (VIDEO_PLACEHOLDER + facts if i == 0 else "") + qt + "<|im_end|>\n"
        text += "<|im_start|>assistant\n" + (THINK if think else "")
        if i < len(answers): text += answers[i] + "<|im_end|>\n"
    return text

class Tok:
    def __init__(self, processor):
        t = processor.tokenizer
        self.letter_ids = [t.convert_tokens_to_ids(L) for L in LETTERS]
        self.im_end = t.convert_tokens_to_ids("<|im_end|>")
        assert len(set(self.letter_ids)) == 4 and all(i is not None and i >= 0 for i in self.letter_ids)

KEEP = ("input_ids", "pixel_values_videos", "video_grid_thw", "mm_token_type_ids")

def encode_pack(processor, tok, frames, vmeta, qtexts, answers, think=False, facts=""):
    """Returns (dict of model inputs, logit positions [k], target indices 0..3 [k])."""
    text = build_text(qtexts, answers, think, facts)
    enc = processor(text=[text], videos=[frames], video_metadata=[vmeta], do_sample_frames=False, return_tensors="pt")
    ids = enc["input_ids"][0].tolist()
    pos = [p for p in range(len(ids) - 1) if ids[p] in tok.letter_ids and ids[p + 1] == tok.im_end]
    assert len(pos) == len(answers), (len(pos), len(answers))
    tgt = [tok.letter_ids.index(ids[p]) for p in pos]
    assert [LETTERS[i] for i in tgt] == list(answers), (tgt, answers)
    inputs = {k: enc[k] for k in KEEP if k in enc}
    return inputs, torch.tensor([p - 1 for p in pos]), torch.tensor(tgt)

def per_task_accuracy(rows):
    """rows: iterable of dicts with 'task', 'pred', 'gt'."""
    import collections
    c, n = collections.Counter(), collections.Counter()
    for r in rows:
        if r.get("gt"): n[r["task"]] += 1; c[r["task"]] += int(r["pred"] == r["gt"])
    out = {t: (c[t] / n[t], n[t]) for t in n}
    tot = sum(n.values())
    out["OVERALL"] = (sum(c.values()) / max(tot, 1), tot)
    return out

def fmt_acc(acc):
    lines = [f"{t:35s} {a:.4f} (n={n})" for t, (a, n) in acc.items() if t != "OVERALL"]
    return "\n".join(sorted(lines)) + f"\n{'OVERALL':35s} {acc['OVERALL'][0]:.4f} (n={acc['OVERALL'][1]})"
