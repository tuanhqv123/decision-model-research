"""CPU-only verification of the generative arm's input pipeline, before any GPU time is spent.

Checks, on real rows:
  1. the rendered prompt starts with exactly one BOS and ends with `<|im_start|>assistant\n<think></think>`;
  2. the chat template's OWN rendering of [user, assistant(answer)] equals prompt + answer + `<|im_end|>\n`, and its
     tokenization starts with exactly our (prompt ids + answer ids) — so the loss mask covers the answer and the
     end-of-turn token and nothing else;
  3. image placeholder count == image_grid_thw.prod() / merge_size**2;
  4. token counts per example, and for EVERY row of the given splits (text tokenized; image tokens from the
     processor's own counter at the size train_d1.load_image produces);
  5. (--forward) one forward: our answer-only loss == the model's own `labels=` loss;
  6. (--backward) one backward under non-reentrant gradient checkpointing with the embeddings frozen: gradients
     reach the vision tower, the projector and the language model, and the embedding gets none;
  7. (--generate) one greedy decode of the BASE model: raw text, token count, greedy == argmax at every step.

Run with CUDA hidden:  CUDA_VISIBLE_DEVICES="" nice -n 19 python benchmarks/gen/cpu_check.py --model <dir> --data <dir>
"""
import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "train"))
from gen_common import END_OF_TURN, messages, parse_action, target_text, user_text  # noqa: E402
from gen_model import (DEFAULT_MAX_NEW_TOKENS, GenEngine, answer_ids, answer_loss, build_example,  # noqa: E402
                       prompt_string, to_device)


def _read(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _q(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def resized(W, H, width=None):
    """The size train_d1.load_image gives a W x H screenshot."""
    width = width or int(os.environ.get("D1_IMG_W", "720"))
    return (width, round(H * width / W)) if W != width else (W, H)


def check_example(processor, rec, data):
    from train_d1 import load_image
    tok = processor.tokenizer
    img = load_image(os.path.join(data, rec["img"]))
    ans = target_text(rec)
    prompt = prompt_string(processor, rec["goal"], rec.get("history"))
    full = processor.apply_chat_template(
        messages(rec["goal"], rec.get("history")) + [{"role": "assistant", "content": ans}],
        enable_thinking=False, tokenize=False)
    ex, labels, n_prompt = build_example(processor, rec["goal"], rec.get("history"), img, ans)
    ids = ex["input_ids"][0].tolist()
    full_ids = processor(text=[full], images=[img], return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    img_id = tok.convert_tokens_to_ids("<|image|>")
    merge = processor.image_processor.merge_size
    n_img = int(ex["image_grid_thw"].prod()) // merge ** 2
    supervised = [t for t in labels[0].tolist() if t != -100]
    r = {
        "ep_step": [rec["ep"], rec["step"]], "action": rec["action"]["action_type"], "W_H": [rec["W"], rec["H"]],
        "image_in": list(img.size), "image_grid_thw": ex["image_grid_thw"][0].tolist(),
        "tokens_total": len(ids), "tokens_prompt": n_prompt, "tokens_image": n_img, "tokens_answer": len(supervised),
        "history_steps": len(rec.get("history") or []),
        "ok_one_bos": ids[0] == tok.bos_token_id and ids.count(tok.bos_token_id) == 1,
        "ok_prompt_tail": prompt.endswith("<|im_start|>assistant\n<think></think>"),
        "ok_full_render": full == prompt + ans + END_OF_TURN + "\n",
        "ok_full_tokens_prefix": full_ids[:len(ids)].tolist() == ids and len(full_ids) >= len(ids),
        "full_render_extra_tokens": len(full_ids) - len(ids),
        "ok_image_placeholders": ids.count(img_id) == n_img,
        "ok_mask_is_answer": supervised == answer_ids(processor, ans)
        and tok.decode(supervised, skip_special_tokens=False) == ans + END_OF_TURN,
        "ok_mask_starts_after_prompt": labels[0, :n_prompt].eq(-100).all().item()
        and labels[0, n_prompt:].ne(-100).all().item(),
        "answer": ans,
    }
    return r, (ex, labels, n_prompt, img)


def token_stats(processor, rows, label):
    """Exact text tokens per row; image tokens from the processor's own counter at load_image's size."""
    tok = processor.tokenizer
    cache, totals, answers, prompts = {}, [], [], []
    for rec in rows:
        w, h = resized(rec["W"], rec["H"])
        if (w, h) not in cache:
            cache[(w, h)] = processor._get_num_multimodal_tokens(image_sizes=[(h, w)])["num_image_tokens"][0]
        text_ids = tok(prompt_string(processor, rec["goal"], rec.get("history")), add_special_tokens=False)["input_ids"]
        n_prompt = len(text_ids) - 1 + cache[(w, h)]  # one <|image|> placeholder becomes n image tokens
        n_ans = len(answer_ids(processor, target_text(rec)))
        prompts.append(n_prompt)
        answers.append(n_ans)
        totals.append(n_prompt + n_ans)
    return {"split": label, "rows": len(rows), "image_tokens_by_size": {f"{k[0]}x{k[1]}": v for k, v in cache.items()},
            "total_tokens": {"p50": _q(totals, .5), "p95": _q(totals, .95), "max": max(totals)},
            "prompt_tokens": {"p50": _q(prompts, .5), "max": max(prompts)},
            "answer_tokens": {"p50": _q(answers, .5), "p95": _q(answers, .95), "max": max(answers)}}


def roundtrip(model, processor, a, rec, img, base_text):
    """train_gen.save -> gen_common.check_contract -> reload -> same greedy text -> one /v1/decide in-process."""
    import base64
    import importlib.util
    import io
    import shutil
    from gen_common import check_contract
    from gen_model import load
    from train_gen import save
    out = {}
    try:
        save(model, processor, a.roundtrip, a.model, {"cpu_check": True})
        out["files"] = sorted(os.listdir(a.roundtrip))
        out["contract"] = check_contract(a.roundtrip)
        model2, proc2 = load(a.roundtrip, "cpu", getattr(torch, a.dtype))
        g2 = GenEngine(model2.eval(), proc2, max_new_tokens=48).run(rec["goal"], rec.get("history"), img)
        out["reloaded_text"] = g2["text"]
        out["same_text_as_before_save"] = g2["text"] == base_text
        del model2
        os.environ.update(GEN_MODEL=a.roundtrip, GEN_DEVICE="cpu", D1_API_TOKEN="cpu-check")
        path = os.path.join(HERE, "..", "..", "serving", "gen_api.py")
        spec = importlib.util.spec_from_file_location("gen_api_cpu_check", path)
        api = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(api)
        from fastapi.testclient import TestClient
        with open(os.path.join(a.data, rec["img"]), "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()
        with TestClient(api.app) as client:  # runs startup: contract check, load, warm-up decode
            health = client.get("/health").json()
            r = client.post("/v1/decide", json={"goal": rec["goal"], "history": rec.get("history") or [],
                                                "image_b64": b64, "track": "V"},
                            headers={"x-api-key": "cpu-check"})
        body = r.json()
        out["api"] = {"status": r.status_code, "health_contract": health.get("contract"),
                      "warmup_s": health.get("warmup_s"),
                      "response": {k: body.get(k) for k in ("action", "point", "text", "app_name", "direction",
                                                            "confidence", "raw", "parse_error",
                                                            "tokens_generated", "latency_ms")}}
        out["api"]["same_text_as_engine"] = body.get("raw") == g2["text"]
        out["ok"] = (out["contract"]["name"] == "gen_v1" and out["same_text_as_before_save"]
                     and r.status_code == 200 and out["api"]["same_text_as_engine"])
    finally:
        shutil.rmtree(a.roundtrip, ignore_errors=True)
    return out


def pick(rows, data, k_per_type=1):
    """One row per action type whose image is on disk, plus the longest history among those."""
    have = [r for r in rows if os.path.exists(os.path.join(data, r["img"]))]
    out, seen = [], set()
    for r in have:
        if r["action"]["action_type"] not in seen:
            out.append(r)
            seen.add(r["action"]["action_type"])
    longest = max(have, key=lambda r: len(r.get("history") or []), default=None)
    if longest is not None and longest not in out:
        out.append(longest)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="dir with compact_<split>.jsonl and imgs/")
    ap.add_argument("--splits", default="cur15k_v1,validation_v1")
    ap.add_argument("--forward", action="store_true")
    ap.add_argument("--backward", action="store_true")
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default="")
    ap.add_argument("--roundtrip", default="", help="(with --generate) save the loaded weights here with "
                    "train_gen.save, reload them, decode again, and serve one request through serving/gen_api.py "
                    "in-process; the directory is deleted afterwards")
    a = ap.parse_args()
    assert not torch.cuda.is_available(), "run with CUDA_VISIBLE_DEVICES='' (CPU only)"
    torch.set_num_threads(a.threads)
    from transformers import AutoProcessor
    import transformers
    report = {"torch": torch.__version__, "transformers": transformers.__version__}
    processor = AutoProcessor.from_pretrained(a.model, trust_remote_code=True)
    tok = processor.tokenizer
    report["special"] = {"bos": [tok.bos_token, tok.bos_token_id], "eos": [tok.eos_token, tok.eos_token_id],
                         "im_end": tok.convert_tokens_to_ids(END_OF_TURN), "pad": [tok.pad_token, tok.pad_token_id]}
    splits = {s: _read(os.path.join(a.data, f"compact_{s}.jsonl")) for s in a.splits.split(",")}
    report["token_stats"] = [token_stats(processor, rows, s) for s, rows in splits.items()]
    examples, keep, first = [], None, None
    for s, rows in splits.items():
        for rec in pick(rows, a.data):
            first = first or rec
            r, ex = check_example(processor, rec, a.data)
            r["split"] = s
            examples.append(r)
            if keep is None or rec["action"]["action_type"] == "type_text":
                keep = (rec, ex)
    if not examples:
        raise SystemExit("no row has its image under --data; copy a few imgs/<ep>/<step>.jpg first")
    report["examples"] = examples
    report["all_ok"] = all(v for r in examples for k, v in r.items() if k.startswith("ok_"))
    report["example_prompt_string"] = prompt_string(processor, first["goal"], first.get("history"))
    print(json.dumps(report, indent=1, ensure_ascii=False), flush=True)

    if a.forward or a.backward or a.generate:
        from transformers import AutoModelForImageTextToText
        dtype = getattr(torch, a.dtype)
        t0 = time.time()
        model = AutoModelForImageTextToText.from_pretrained(a.model, trust_remote_code=True, dtype=dtype)
        n_params = sum(p.numel() for p in model.parameters())
        emb = model.get_input_embeddings().weight
        report["model"] = {"load_s": round(time.time() - t0, 1), "params_B": round(n_params / 1e9, 3),
                           "embedding_params_B": round(emb.numel() / 1e9, 3),
                           "tied_lm_head": model.get_output_embeddings().weight is emb,
                           "trainable_if_embed_frozen_B": round((n_params - emb.numel()) / 1e9, 3)}
        rec, (ex, labels, n_prompt, img) = keep
        exd = to_device(ex, "cpu", dtype)
        if a.forward:
            model.eval()
            with torch.no_grad():
                t0 = time.time()
                loss, st = answer_loss(model, exd, labels, n_prompt)
                ref = model(**exd, labels=labels, use_cache=False)
                report["forward"] = {"ep_step": [rec["ep"], rec["step"]], "answer_loss": round(float(loss), 4),
                                     "hf_labels_loss": round(float(ref.loss), 4),
                                     "ok_losses_agree": abs(float(loss) - float(ref.loss)) < 1e-3,
                                     "full_logits_shape": list(ref.logits.shape), **st,
                                     "seconds": round(time.time() - t0, 1)}
            print("FORWARD " + json.dumps(report["forward"]), flush=True)
        if a.backward:
            model.train()
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            for p in model.parameters():
                p.requires_grad_(False)
            probes = {}
            for name, p in model.named_parameters():
                for key in ("vision_tower", "multi_modal_projector", "language_model"):
                    if key in name and key not in probes and p.numel() < 300_000:
                        probes[key] = (name, p)
            last_norm = [(n, p) for n, p in model.named_parameters() if "language_model" in n and "norm" in n][-1]
            probes["language_model_last_norm"] = last_norm
            for _, p in probes.values():
                p.requires_grad_(True)
            t0 = time.time()
            loss, _ = answer_loss(model, exd, labels, n_prompt)
            loss.backward()
            report["backward"] = {"loss": round(float(loss), 4), "seconds": round(time.time() - t0, 1),
                                  "embedding_grad_is_none": emb.grad is None,
                                  "grad_norms": {k: [n, float(p.grad.norm()) if p.grad is not None else None]
                                                 for k, (n, p) in probes.items()}}
            report["backward"]["ok_grads_reach_all_probes"] = all(
                v[1] is not None and v[1] > 0 for v in report["backward"]["grad_norms"].values())
            print("BACKWARD " + json.dumps(report["backward"]), flush=True)
            model.gradient_checkpointing_disable()
            model.zero_grad(set_to_none=True)
        if a.generate:
            model.eval()
            eng = GenEngine(model, processor, max_new_tokens=min(DEFAULT_MAX_NEW_TOKENS, 48))
            g = eng.run(rec["goal"], rec.get("history"), img)
            parsed, err, flags = parse_action(g["text"])
            report["generate_base_zero_shot"] = {
                "ep_step": [rec["ep"], rec["step"]], "gold": target_text(rec), "raw": g["text"],
                "tokens_generated": g["tokens_generated"], "hit_eos": g["hit_eos"],
                "greedy_consistent": g["greedy_consistent"], "parsed": parsed, "parse_error": err, "flags": flags,
                "seconds": round(g["model_ms"] / 1000, 1)}
            print("GENERATE " + json.dumps(report["generate_base_zero_shot"], ensure_ascii=False), flush=True)
            if a.roundtrip:
                report["roundtrip"] = roundtrip(model, processor, a, rec, img, g["text"])
                print("ROUNDTRIP " + json.dumps(report["roundtrip"], ensure_ascii=False), flush=True)
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(report, fh, indent=1, ensure_ascii=False)
    ok = report["all_ok"] and report.get("forward", {}).get("ok_losses_agree", True) and \
        report.get("backward", {}).get("ok_grads_reach_all_probes", True) and \
        report.get("backward", {}).get("embedding_grad_is_none", True) and \
        report.get("roundtrip", {}).get("ok", True)
    print(f"CPU_CHECK_{'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
