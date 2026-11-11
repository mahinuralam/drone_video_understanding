"""Letter log-prob evaluation with per-video prefix caching (system + video computed once per video,
each question suffix appended and then reverted). Several option permutations (--shifts) are evaluated
per prefix pass (the prefix is ~77% of the cost). Dumps per-question per-option probabilities in ORIGINAL
letter order, one file per shift, written atomically (tmp + rename) so a partial dump never looks complete.
Usage:
  eval_probs.py --split splits/dev.json [--adapter ckpt/run1_9b/epoch_2] --grid 0 --shift 0 --out results/x.jsonl
  eval_probs.py ... --grid 0 --shifts 0,1,2,3 --out_prefix results/run1_9b_e2_dev_g0   # -> results/run1_9b_e2_dev_g0_s{0,1,2,3}.jsonl
"""
import argparse, json, os, sys, time, collections, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, torch.nn.functional as F
from transformers import AutoProcessor, AutoModelForImageTextToText, DynamicCache
import reva_common as C

# Pre-Ampere GPUs (e.g. Quadro RTX 8000, sm_75): SDPA with enable_gqa falls back to the math kernel (7x slower, O(N^2) memory);
# repeating the KV heads instead lets the memory-efficient cutlass kernel run. No effect on sm_80+.
if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] < 8:
    import transformers.integrations.sdpa_attention as _sa
    _sa.use_gqa_in_sdpa = lambda *a, **k: False
    print("[sdpa] pre-Ampere GPU: GQA-in-SDPA disabled (KV heads repeated -> memory-efficient kernel)", flush=True)

@torch.inference_mode()
def evaluate(model, processor, tok, rows, grid=0, shift=0, think=False, out_path=None, log_every=100, tag="",
             shifts=None, out_paths=None, long_frames=0, long_sec=8.0, facts_by_vid=None, pix_dtype=torch.bfloat16):
    """shifts/out_paths (dict shift -> path) generalise shift/out_path. Returns (acc, res) for a single shift,
    else {shift: (acc, res)}."""
    single = shifts is None
    if single: shifts = (shift,); out_paths = {shift: out_path}
    model.eval()
    by_vid = collections.OrderedDict()
    for q in rows: by_vid.setdefault(q["video_path"], []).append(q)
    perms = {s: C.cyclic_perm(s) for s in shifts}
    fos = {s: open(out_paths[s] + ".tmp", "w") for s in shifts if out_paths and out_paths.get(s)}
    res = {s: [] for s in shifts}; t0 = time.time(); nq = 0; t_prefix = 0.0; t_suffix = 0.0
    for vi, (vp, qs) in enumerate(by_vid.items()):
        frames, vmeta = C.load_video(vp, grid, long_frames=long_frames, long_sec=long_sec)
        facts = C.facts_block(facts_by_vid.get(vp, [])) if facts_by_vid is not None else ""
        enc = processor(text=[C.prefix_text(facts)], videos=[frames], video_metadata=[vmeta], do_sample_frames=False, return_tensors="pt")
        inputs = {k: enc[k].cuda() for k in C.KEEP if k in enc}
        inputs["pixel_values_videos"] = inputs["pixel_values_videos"].to(pix_dtype)
        cache = DynamicCache(config=model.config)  # config -> correct layer types (hybrid linear-attention models too)
        t1 = time.time()
        model(**inputs, past_key_values=cache, use_cache=True, logits_to_keep=1)
        L0 = inputs["input_ids"].shape[1]
        torch.cuda.synchronize(); t_prefix += time.time() - t1; t1 = time.time()
        can_crop = all(type(l).__name__ == "DynamicLayer" for l in cache.layers)  # plain KV cache: crop; else snapshot+copy
        snapshot = None if can_crop else copy.deepcopy(cache)
        for q in qs:
            for s in shifts:
                perm = perms[s]
                opts, _ = C.permute_options(q, perm)
                suf = processor.tokenizer(C.suffix_text(C.question_text(q["question"], opts), think),
                                          add_special_tokens=False, return_tensors="pt")["input_ids"].cuda()
                Ls = suf.shape[1]
                c = cache if can_crop else copy.deepcopy(snapshot)
                out = model(input_ids=suf, past_key_values=c, cache_position=torch.arange(L0, L0 + Ls, device="cuda"),
                            use_cache=True, logits_to_keep=1)
                lg = out.logits[0, -1].float()
                top = lg.argmax().item()
                probs = F.softmax(lg[tok.letter_ids], -1).tolist()
                p_orig = {perm[i]: probs[i] for i in range(4)}  # new position i showed original option perm[i]
                pred = max(C.LETTERS, key=lambda L: p_orig[L])  # ties -> lowest letter, independent of the shift
                r = {"qa_id": q["qa_id"], "task": q["task"], "video_path": vp, "grid": grid, "shift": s, "pred": pred,
                     "gt": q.get("correct_answer", ""), "probs": [p_orig[L] for L in C.LETTERS],
                     "top_is_letter": top in tok.letter_ids, "n_prefix": L0, "n_suffix": Ls}
                res[s].append(r)
                if s in fos: fos[s].write(json.dumps(r) + "\n")
                if can_crop: cache.crop(-Ls)
            nq += 1
        torch.cuda.synchronize(); t_suffix += time.time() - t1
        if (vi + 1) % log_every == 0:
            print(f"[{tag}] {vi+1}/{len(by_vid)} videos {nq} q x {len(shifts)} shifts {time.time()-t0:.0f}s (prefix {t_prefix:.0f}s suffix {t_suffix:.0f}s) acc(s{shifts[0]})={C.per_task_accuracy(res[shifts[0]])['OVERALL'][0]:.4f}", flush=True)
    for s, fo in fos.items(): fo.close(); os.replace(out_paths[s] + ".tmp", out_paths[s])
    out = {}
    for s in shifts:
        acc = C.per_task_accuracy(res[s]); out[s] = (acc, res[s])
        frac_letter = sum(r["top_is_letter"] for r in res[s]) / max(len(res[s]), 1)
        print(f"[{tag}] done {nq} q / {len(by_vid)} videos in {time.time()-t0:.0f}s (prefix {t_prefix:.0f}s, suffix {t_suffix:.0f}s); top1-is-letter {frac_letter:.3f}; grid={grid} shift={s}\n" + C.fmt_acc(acc), flush=True)
    return out[shifts[0]] if single else out

def linear_attn_fp32(model, out_dtype=torch.float16):
    """fp16 GPUs (Turing): the hybrid model's GatedDeltaNet layers overflow in fp16 (non-finite grads every step).
    Run those modules in fp32 with casting hooks; the rest of the model stays fp16. Returns the number of patched modules."""
    n = 0
    def pre(mod, args, kwargs):
        args = tuple(a.float() if torch.is_tensor(a) and a.is_floating_point() else a for a in args)
        kwargs = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in kwargs.items()}
        return args, kwargs
    def post(mod, args, out):
        if torch.is_tensor(out): return out.to(out_dtype) if out.is_floating_point() else out
        if isinstance(out, tuple): return tuple(o.to(out_dtype) if torch.is_tensor(o) and o.is_floating_point() else o for o in out)
        return out
    for name, m in model.named_modules():
        if type(m).__name__.endswith("GatedDeltaNet"):
            m.float(); m.register_forward_pre_hook(pre, with_kwargs=True); m.register_forward_hook(post); n += 1
    return n

def model_kwargs(load_4bit):
    kw = {}
    if load_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                                      bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_skip_modules=["model.visual", "lm_head"])
    return kw

def load_model(name, adapter=None, load_4bit=False, fp16=False, vision_fp32=False, lin_fp32=False):
    model = AutoModelForImageTextToText.from_pretrained(name, dtype=torch.float16 if fp16 else torch.bfloat16, device_map="cuda", attn_implementation="sdpa", **model_kwargs(load_4bit))
    if vision_fp32: model.model.visual.float()  # Turing (fp16-only) safety: run the frozen vision tower in fp32
    if lin_fp32: print(f"[fp16] GatedDeltaNet layers in fp32: {linear_attn_fp32(model)}", flush=True)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    return model.eval()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--split", default="splits/dev.json")
    ap.add_argument("--grid", type=int, default=0)
    ap.add_argument("--shift", type=int, default=0)
    ap.add_argument("--shifts", default="", help="comma list; with --out_prefix writes <prefix>_s<shift>.jsonl per shift")
    ap.add_argument("--out", default="")
    ap.add_argument("--out_prefix", default="")
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--load_4bit", action="store_true")
    ap.add_argument("--fp16", action="store_true", help="fp16 base weights (GPUs without bf16, e.g. sm_75)")
    ap.add_argument("--vision_fp32", action="store_true", help="run the vision tower in fp32 (with --fp16)")
    ap.add_argument("--linear_attn_fp32", action="store_true", help="run GatedDeltaNet layers in fp32 (Qwen3.5 under --fp16)")
    ap.add_argument("--long_frames", type=int, default=0, help="use the <long_frames>-frame cache for clips longer than --long_sec")
    ap.add_argument("--long_sec", type=float, default=8.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--facts_json", default="", help="Format B: put the QAs of the same video from this split (e.g. splits/train_run1.json) into the prompt as known facts")
    ap.add_argument("--tag", default="eval")
    args = ap.parse_args()
    os.chdir(C.ROOT)
    processor = AutoProcessor.from_pretrained(args.model); tok = C.Tok(processor)
    rows = json.load(open(args.split))["QA"]
    if args.limit: rows = rows[:args.limit]
    facts_by_vid = None
    if args.facts_json:
        facts_by_vid = collections.defaultdict(list)
        for q in json.load(open(args.facts_json))["QA"]: facts_by_vid[q["video_path"]].append(q)
        print(f"facts from {args.facts_json}: {sum(len(v) for v in facts_by_vid.values())} QAs over {len(facts_by_vid)} videos", flush=True)
    model = load_model(args.model, args.adapter or None, args.load_4bit, args.fp16, args.vision_fp32, args.linear_attn_fp32)
    PIX = torch.float32 if args.vision_fp32 else (torch.float16 if args.fp16 else torch.bfloat16)
    print(f"loaded {args.model} adapter={args.adapter!r} 4bit={args.load_4bit} fp16={args.fp16} vision_fp32={args.vision_fp32}; gpu {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)
    if args.shifts:
        shifts = tuple(int(s) for s in args.shifts.split(","))
        assert args.out_prefix, "--shifts needs --out_prefix"
        out_paths = {s: f"{args.out_prefix}_s{s}.jsonl" for s in shifts}
        os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)
        evaluate(model, processor, tok, rows, args.grid, think=args.think, tag=args.tag, shifts=shifts, out_paths=out_paths,
                 long_frames=args.long_frames, long_sec=args.long_sec, facts_by_vid=facts_by_vid, pix_dtype=PIX)
    else:
        assert args.out, "--out required"
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        evaluate(model, processor, tok, rows, args.grid, args.shift, args.think, args.out, tag=args.tag,
                 long_frames=args.long_frames, long_sec=args.long_sec, facts_by_vid=facts_by_vid, pix_dtype=PIX)
