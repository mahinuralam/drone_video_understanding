"""Packed LoRA fine-tuning for ReVA MCQ (Qwen3-VL-8B-Instruct by default; Qwen3.5-9B with --think).
One sequence = system + video (32 cached frames) + k QA turns of the same video, k ~ U{1..kmax},
re-partitioned every epoch (deterministic in seed+epoch so resume is exact). Loss only on the
answer-letter token; logits gathered at label positions only. Vision tower + merger frozen (no_grad),
LoRA on q/k/v/o/gate/up/down of all LLM layers, gradient checkpointing, micro-batch 1 x grad-accum.
"""
import argparse, json, os, sys, time, math, random, collections, glob, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoProcessor, AutoModelForImageTextToText, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel
import reva_common as C

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
ap.add_argument("--train", default="splits/train_run1.json")
ap.add_argument("--dev", default="splits/dev.json")
ap.add_argument("--out", required=True)
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--grad_accum", type=int, default=4)
ap.add_argument("--warmup", type=float, default=0.03)
ap.add_argument("--r", type=int, default=32); ap.add_argument("--alpha", type=int, default=64); ap.add_argument("--dropout", type=float, default=0.05)
ap.add_argument("--kmax", type=int, default=6)
ap.add_argument("--grids", default="0,1,2", help="frame-offset grids sampled per pack")
ap.add_argument("--max_steps", type=int, default=0, help="optimizer steps; 0 = full schedule")
ap.add_argument("--save_every", type=int, default=100)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--think", action="store_true", help="Qwen3.5 non-thinking template (empty think block)")
ap.add_argument("--no_perm_aug", action="store_true", help="disable random option permutation per question per epoch")
ap.add_argument("--weights", default="Change Detection:2.0,Temporal Grounding:1.4,Object and Land Cover Recognition:1.4,Geometric Relation:0.6,Structural Layout:0.6")
ap.add_argument("--eval_dev", type=int, default=1)
ap.add_argument("--eval_grid", type=int, default=0)
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--resume", action="store_true")
ap.add_argument("--unfreeze_merger", action="store_true")
ap.add_argument("--load_4bit", action="store_true", help="nf4 base weights (QLoRA) for models that do not fit in bf16")
ap.add_argument("--fp16", action="store_true", help="fp16 base weights + GradScaler (GPUs without bf16, e.g. RTX 8000 sm_75); LoRA and loss stay fp32")
ap.add_argument("--vision_fp32", action="store_true", help="run the frozen vision tower in fp32 (fp16 overflow safety)")
ap.add_argument("--linear_attn_fp32", action="store_true", help="run GatedDeltaNet layers in fp32 (Qwen3.5 under --fp16; fixes non-finite grads)")
ap.add_argument("--long_frames", type=int, default=0, help="use the <long_frames>-frame cache for clips longer than --long_sec")
ap.add_argument("--long_sec", type=float, default=8.0)
ap.add_argument("--facts", action="store_true", help="Format B: put the other training QAs of the same video (not the pack's own questions) into the prompt as known facts; epoch-end evals use the facts of the training split")
ap.add_argument("--max_facts", type=int, default=40)
ap.add_argument("--target_regex", default=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")
ap.add_argument("--log_every", type=int, default=5)
ap.add_argument("--test_split", default="", help="if set, run test inference (grid eval_grid, shift 0) at the end of listed epochs")
ap.add_argument("--test_at_epochs", default="0,1,2")
args = ap.parse_args()
os.chdir(C.ROOT); os.makedirs(args.out, exist_ok=True)
json.dump(vars(args), open(args.out + "/args.json", "w"), indent=1)
torch.manual_seed(args.seed); random.seed(args.seed)

def parse_weights(s):
    w = {}
    for part in s.split(","):
        if part.strip(): k, v = part.rsplit(":", 1); w[k.strip()] = float(v)
    return w

def build_epoch_packs(rows, epoch, seed, kmax, weights, grids, perm_aug):
    rng = random.Random(seed * 1000 + epoch)
    by_vid = collections.defaultdict(list)
    for q in rows:
        w = weights.get(q["task"], 1.0); m = int(w) + (1 if rng.random() < w - int(w) else 0)
        for _ in range(m): by_vid[q["video_path"]].append(q)
    packs = []
    for vp in sorted(by_vid):
        items = by_vid[vp][:]; rng.shuffle(items)
        while items:
            k = rng.randint(1, kmax); chunk, rest = [], []
            for q in items:
                if len(chunk) < k and all(q["qa_id"] != c["qa_id"] for c in chunk): chunk.append(q)
                else: rest.append(q)
            items = rest
            perms = [tuple(rng.sample(C.LETTERS, 4)) if perm_aug else tuple(C.LETTERS) for _ in chunk]
            packs.append({"video_path": vp, "qs": chunk, "perms": perms, "grid": rng.choice(grids)})
    rng.shuffle(packs)
    return packs

class PackDataset(Dataset):
    def __init__(self, packs, processor, tok, think, long_frames=0, long_sec=8.0, facts_by_vid=None, max_facts=40):
        self.packs, self.processor, self.tok, self.think, self.long_frames, self.long_sec = packs, processor, tok, think, long_frames, long_sec
        self.facts_by_vid, self.max_facts = facts_by_vid, max_facts
    def __len__(self): return len(self.packs)
    def __getitem__(self, i):
        p = self.packs[i]
        frames, vmeta = C.load_video(p["video_path"], p["grid"], long_frames=self.long_frames, long_sec=self.long_sec)
        qtexts, answers = [], []
        for q, perm in zip(p["qs"], p["perms"]):
            opts, ans = C.permute_options(q, perm); qtexts.append(C.question_text(q["question"], opts)); answers.append(ans)
        facts = ""
        if self.facts_by_vid is not None:
            own = {q["qa_id"] for q in p["qs"]}
            facts = C.facts_block([q for q in self.facts_by_vid.get(p["video_path"], []) if q["qa_id"] not in own], self.max_facts)
        inputs, pos, tgt = C.encode_pack(self.processor, self.tok, frames, vmeta, qtexts, answers, self.think, facts)
        return {"inputs": inputs, "pos": pos, "tgt": tgt, "idx": i, "tasks": [q["task"] for q in p["qs"]]}

processor = AutoProcessor.from_pretrained(args.model); tok = C.Tok(processor)
rows = json.load(open(args.train))["QA"]
dev_rows = json.load(open(args.dev))["QA"] if args.dev and args.eval_dev else []
test_rows = json.load(open(args.test_split))["QA"] if args.test_split else []
facts_by_vid = None
if args.facts:
    facts_by_vid = collections.defaultdict(list)
    for q in rows: facts_by_vid[q["video_path"]].append(q)
    print(f"Format B: facts from the training split, {len(rows)} QAs over {len(facts_by_vid)} videos (pack's own questions excluded)", flush=True)
test_at = {int(e) for e in args.test_at_epochs.split(",") if e.strip()}
weights = parse_weights(args.weights); grids = [int(g) for g in args.grids.split(",")]
packs_per_epoch = [build_epoch_packs(rows, e, args.seed, args.kmax, weights, grids, not args.no_perm_aug) for e in range(args.epochs)]
n_micro_total = sum(len(p) for p in packs_per_epoch)
total_steps = args.max_steps or sum(math.ceil(len(p) / args.grad_accum) for p in packs_per_epoch)  # partial accumulation is flushed at each epoch end
print(f"train rows {len(rows)}; packs/epoch {[len(p) for p in packs_per_epoch]}; questions/epoch {[sum(len(x['qs']) for x in p) for p in packs_per_epoch]}; optimizer steps {total_steps}", flush=True)

t0 = time.time()
from eval_probs import evaluate, model_kwargs, linear_attn_fp32
DT = torch.float16 if args.fp16 else torch.bfloat16
model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=DT, device_map="cuda", attn_implementation="sdpa", **model_kwargs(args.load_4bit))
model.config.use_cache = False
if args.vision_fp32: model.model.visual.float()
if args.linear_attn_fp32: print(f"[fp16] GatedDeltaNet layers in fp32: {linear_attn_fp32(model, DT)}", flush=True)
PIX = torch.float32 if args.vision_fp32 else DT
for p in model.parameters(): p.requires_grad_(False)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

def latest_ckpt(out):
    cks = [d for d in glob.glob(out + "/step_*") + glob.glob(out + "/epoch_*") if os.path.exists(d + "/state.pt")]
    return max(cks, key=lambda d: torch.load(d + "/state.pt", map_location="cpu", weights_only=False)["micro"]) if cks else None

merger_names = [n for n, m in model.named_modules() if "PatchMerger" in type(m).__name__]  # visual.merger + deepstack mergers
if args.unfreeze_merger: print("unfreezing mergers:", merger_names, flush=True)
ckpt = latest_ckpt(args.out) if args.resume else None
if ckpt:
    model = PeftModel.from_pretrained(model, ckpt, is_trainable=True); print("resumed adapter from", ckpt, flush=True)
else:
    cfg = LoraConfig(r=args.r, lora_alpha=args.alpha, lora_dropout=args.dropout, bias="none",
                     target_modules=args.target_regex,
                     modules_to_save=merger_names if args.unfreeze_merger else None)
    model = get_peft_model(model, cfg)
model.print_trainable_parameters()
if args.unfreeze_merger:
    # PEFT keeps modules_to_save copies in the base dtype (bf16); AdamW updates below half an ULP would round to zero.
    # Run the unfrozen merger copies in fp32 (bf16 in/out) before the optimizer is built so Adam moments are fp32 too.
    from peft.utils.other import ModulesToSaveWrapper
    n_up = 0
    for m in model.modules():
        if isinstance(m, ModulesToSaveWrapper):
            act = m.active_adapters; act = act[0] if isinstance(act, (list, tuple)) else act
            sub = m.modules_to_save[act]; sub.float()
            sub.register_forward_pre_hook(lambda mod, a: (a[0].float(),) + tuple(a[1:]))
            sub.register_forward_hook(lambda mod, a, o: o.to(DT))
            n_up += 1
    print(f"unfrozen merger copies run in fp32: {n_up}", flush=True)
inner = model.base_model.model.model  # Qwen3VLModel (or Qwen3_5Model)
if not args.unfreeze_merger and hasattr(inner, "get_video_features"):
    _orig_gvf = inner.get_video_features
    def _gvf(*a, **k):
        with torch.no_grad(): return _orig_gvf(*a, **k)
    inner.get_video_features = _gvf
letter_vocab = torch.tensor(tok.letter_ids, device="cuda")
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.999))
sched = get_cosine_schedule_with_warmup(opt, int(round(args.warmup * total_steps)), total_steps)
scaler = torch.amp.GradScaler("cuda", enabled=args.fp16)
nonfinite = 0; skipped_steps = 0
start_epoch = start_pack = opt_step = micro = 0
if ckpt:
    st = torch.load(ckpt + "/state.pt", map_location="cpu", weights_only=False)
    opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
    if "scaler" in st and args.fp16: scaler.load_state_dict(st["scaler"])
    start_epoch, start_pack, opt_step, micro = st["epoch"], st["pack_idx"], st["opt_step"], st["micro"]
    torch.set_rng_state(st["rng"]); torch.cuda.set_rng_state(st["cuda_rng"]); random.setstate(st["py_rng"])
    print(f"resumed at epoch {start_epoch} pack {start_pack} opt_step {opt_step} micro {micro}", flush=True)
print(f"model ready in {time.time()-t0:.0f}s; gpu {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

def save_ckpt(name, epoch, pack_idx):
    d = f"{args.out}/{name}"; os.makedirs(d, exist_ok=True)
    model.save_pretrained(d)
    torch.save({"opt": opt.state_dict(), "sched": sched.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch, "pack_idx": pack_idx, "opt_step": opt_step,
                "micro": micro, "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(), "py_rng": random.getstate()}, d + "/state.pt")
    if name.startswith("step_"):
        olds = sorted(glob.glob(args.out + "/step_*"), key=lambda x: int(x.rsplit("_", 1)[1]))[:-2]
        for o in olds: shutil.rmtree(o, ignore_errors=True)
    print(f"saved {d}", flush=True)

log = open(args.out + "/train_log.jsonl", "a")
ema_loss = ema_acc = None; done = False
seq_times, gpu_times, tok_counts = [], [], []
task_loss = collections.defaultdict(list)
model.train()
for epoch in range(start_epoch, args.epochs):
    packs = packs_per_epoch[epoch]
    ds = PackDataset(packs, processor, tok, args.think, args.long_frames, args.long_sec, facts_by_vid, args.max_facts)
    loader = DataLoader(Subset(ds, range(start_pack, len(packs))), batch_size=None, shuffle=False, num_workers=args.workers, prefetch_factor=4 if args.workers else None, persistent_workers=False)
    pack_idx = start_pack; t_prev = time.time()
    for batch in loader:
        inputs = {k: v.cuda(non_blocking=True) for k, v in batch["inputs"].items()}
        inputs["pixel_values_videos"] = inputs["pixel_values_videos"].to(PIX)
        pos = batch["pos"].cuda(); tgt = batch["tgt"].cuda()
        torch.cuda.synchronize(); t1 = time.time()
        out = model(**inputs, logits_to_keep=pos, use_cache=False)
        logits = out.logits[0].float()  # [k, vocab] at the positions that predict the answer letters
        losses = F.cross_entropy(logits, letter_vocab[tgt], reduction="none")  # full-vocab CE on the letter token
        loss = losses.mean()
        if not torch.isfinite(loss):
            nonfinite += 1; print(f"WARNING non-finite loss at micro {micro} (pack {pack_idx}, {nonfinite} so far); batch skipped", flush=True)
            if nonfinite > 20: raise RuntimeError("too many non-finite losses")
            micro += 1; pack_idx += 1; t_prev = time.time(); continue
        scaler.scale(loss / args.grad_accum).backward()
        torch.cuda.synchronize(); t2 = time.time()
        gpu_times.append(t2 - t1); seq_times.append(t2 - t_prev); t_prev = t2; tok_counts.append(inputs["input_ids"].shape[1])
        for t, l in zip(batch["tasks"], losses.tolist()): task_loss[t].append(l)
        acc = (logits[:, letter_vocab].argmax(-1) == tgt).float().mean().item(); lv = loss.item()  # 4-way letter accuracy
        ema_loss = lv if ema_loss is None else 0.98 * ema_loss + 0.02 * lv
        ema_acc = acc if ema_acc is None else 0.98 * ema_acc + 0.02 * acc
        micro += 1; pack_idx += 1
        if micro % args.grad_accum == 0:
            scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
            sc0 = scaler.get_scale() if args.fp16 else None
            scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad(set_to_none=True); opt_step += 1
            if args.fp16 and scaler.get_scale() < sc0: skipped_steps += 1
            if opt_step % args.log_every == 0 or opt_step <= 5:
                n = args.log_every * args.grad_accum
                rec = {"epoch": epoch, "step": opt_step, "micro": micro, "pack": pack_idx, "loss": round(lv, 4), "ema_loss": round(ema_loss, 4), "ema_acc": round(ema_acc, 4),
                       "grad_norm": round(gn, 3), "lr": sched.get_last_lr()[0], "s_per_seq_wall": round(sum(seq_times[-n:]) / len(seq_times[-n:]), 3),
                       "s_per_seq_gpu": round(sum(gpu_times[-n:]) / len(gpu_times[-n:]), 3), "tokens": round(sum(tok_counts[-n:]) / len(tok_counts[-n:])),
                       "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2), "elapsed_min": round((time.time() - t0) / 60, 1),
                       **({"scale": scaler.get_scale(), "skipped_steps": skipped_steps, "nonfinite": nonfinite} if args.fp16 else {})}
                print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
            if args.save_every and opt_step % args.save_every == 0: save_ckpt(f"step_{opt_step}", epoch, pack_idx)
            if args.max_steps and opt_step >= args.max_steps: done = True; break
    if done: break
    # ---- epoch end: flush a partial accumulation so epoch checkpoints are exactly resumable ----
    rem = micro % args.grad_accum
    if rem:
        for p in params:
            if p.grad is not None: p.grad.mul_(args.grad_accum / rem)
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0); scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad(set_to_none=True); opt_step += 1
        micro += args.grad_accum - rem
    tl = {t: round(sum(v) / len(v), 4) for t, v in task_loss.items()}; task_loss.clear()
    print(f"epoch {epoch} done: step {opt_step}, per-task train loss {json.dumps(tl)}", flush=True)
    save_ckpt(f"epoch_{epoch}", epoch + 1, 0)
    if dev_rows:
        acc, _ = evaluate(model, processor, tok, dev_rows, grid=args.eval_grid, shift=0, think=args.think,
                          out_path=f"{args.out}/dev_epoch{epoch}_g{args.eval_grid}_s0.jsonl", tag=f"dev-e{epoch}",
                          long_frames=args.long_frames, long_sec=args.long_sec, facts_by_vid=facts_by_vid, pix_dtype=PIX)
        rec = {"epoch": epoch, "step": opt_step, "dev": {t: [round(a, 4), n] for t, (a, n) in acc.items()}, "train_task_loss": tl}
        log.write(json.dumps(rec) + "\n"); log.flush()
        model.train()
    if test_rows and epoch in test_at:
        evaluate(model, processor, tok, test_rows, grid=args.eval_grid, shift=0, think=args.think,
                 out_path=f"{args.out}/test_epoch{epoch}_g{args.eval_grid}_s0.jsonl", tag=f"test-e{epoch}",
                 long_frames=args.long_frames, long_sec=args.long_sec, facts_by_vid=facts_by_vid, pix_dtype=PIX)
        model.train()
    start_pack = 0
print(f"finished: opt_step {opt_step}, micro {micro}, mean s/seq wall {sum(seq_times)/max(len(seq_times),1):.2f} gpu {sum(gpu_times)/max(len(gpu_times),1):.2f}, "
      f"mean tokens {sum(tok_counts)/max(len(tok_counts),1):.0f}, peak {torch.cuda.max_memory_allocated()/1e9:.2f} GB, total {(time.time()-t0)/60:.1f} min"
      + (f", fp16: nonfinite losses {nonfinite}, scaler-skipped steps {skipped_steps}, final scale {scaler.get_scale()}" if args.fp16 else ""), flush=True)
if not glob.glob(args.out + "/epoch_*") or done: save_ckpt(f"final_step_{opt_step}", epoch, pack_idx)
