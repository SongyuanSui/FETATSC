import json, os, time, argparse, random, math, re, heapq
from collections import defaultdict
from typing import List, Sequence, Optional, Any, Dict, Tuple
from tqdm import tqdm
import openai
from fastdtw import fastdtw
from myllm import MyDeepSeek
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

_TLS = threading.local()

def get_thread_local_ds(model_name: str) -> MyDeepSeek:
    if not hasattr(_TLS, "ds") or getattr(_TLS, "model_name", None) != model_name:
        _TLS.ds = MyDeepSeek(model_name=model_name)
        _TLS.model_name = model_name
    return _TLS.ds

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='LLM agent TSC (DTW retrieval, multi-agent per-channel, concurrent per sample)'
    )
    parser.add_argument('--train_json', type=str, required=True, help='Processed TRAIN json (values can be 2D: [num_dims][len])')
    parser.add_argument('--test_json', type=str, required=True, help='Processed TEST json (values can be 2D: [num_dims][len])')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save predictions json')
    parser.add_argument('--max_len', type=int, default=384, help='display/downsample length for sequences in prompts')
    parser.add_argument('--decimals', type=int, default=3, help='decimal places for formatting')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--workers', type=int, default=8, help='thread pool size for per-sample per-channel concurrency')
    parser.add_argument('--max_dims', type=int, default=None, help='Use only the first N channels (None = all)')
    parser.add_argument('--eps_other', type=float, default=0.05, help='epsilon vote to non-selected classes per agent')
    parser.add_argument('--clip_lo', type=float, default=0.05, help='clip lower bound for agent confidence')
    parser.add_argument('--clip_hi', type=float, default=0.95, help='clip upper bound for agent confidence')
    parser.add_argument('--neighbors_k', type=int, default=3, help='top-K neighbors by DTW for few-shot (recommended 3)')
    parser.add_argument('--dtw_len', type=int, default=None, help='length used for DTW; default=max_len')
    parser.add_argument('--dtw_radius_frac', type=float, default=0.10, help='fastdtw radius as a fraction of dtw_len')
    parser.add_argument('--dtw_radius', type=int, default=None, help='override radius; if set, ignores dtw_radius_frac')
    parser.add_argument('--z_norm', type=int, default=1, help='1: z-score normalize sequences before DTW; 0: off')
    parser.add_argument('--preselect_k', type=int, default=0, help='if >0: preselect K by Euclidean distance, then run DTW')
    parser.add_argument('--model_name', type=str, default='deepseek-reasoner',
                        help='LLM model name for DeepSeek client (default: deepseek-reasoner)')
    return parser.parse_args()

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def is_1d(arr: Any) -> bool:
    return isinstance(arr, Sequence) and (len(arr) == 0 or not isinstance(arr[0], Sequence))

def ensure_2d(values_any: Any) -> List[List[float]]:
    if values_any is None:
        return [[]]
    if is_1d(values_any):
        return [list(values_any)]
    return [list(dim) for dim in values_any]

def downsample_1d(values: Sequence[float], max_len: Optional[int]) -> List[float]:
    if max_len is None or max_len <= 0 or len(values) <= max_len:
        return list(values)
    n = len(values)
    idxs = [round(i * (n - 1) / (max_len - 1)) for i in range(max_len)]
    return [values[i] for i in idxs]

def z_norm_1d(values: Sequence[float]) -> List[float]:
    n = len(values)
    if n == 0: return []
    mu = sum(values) / n
    var = sum((v - mu) * (v - mu) for v in values) / n
    if var <= 1e-12:
        return [v - mu for v in values]
    sd = math.sqrt(var)
    return [(v - mu) / sd for v in values]

def format_series_1d(values: Sequence[float], decimals=3) -> str:
    return ",".join(f"{v:.{decimals}f}" for v in values)

def euclidean_sq(a: Sequence[float], b: Sequence[float]) -> float:
    return sum((x - y) * (x - y) for x, y in zip(a, b))

def get_classes_from_train(train: List[dict]) -> List[int]:
    return sorted(set(r["label"] for r in train))

def build_per_dim_train_cache(train: List[dict], dtw_len: int, z_norm: bool) -> Dict[int, List[dict]]:
    per_dim: Dict[int, List[dict]] = defaultdict(list)
    for idx, r in enumerate(train):
        vals2d = ensure_2d(r.get("values"))
        label = r.get("label")
        for d, seq in enumerate(vals2d):
            x = downsample_1d(seq, dtw_len)
            if z_norm:
                x = z_norm_1d(x)
            per_dim[d].append({
                "id": r.get("id", idx),
                "label": label,
                "seq": x
            })
    return per_dim

def fastdtw_distance(a: Sequence[float], b: Sequence[float], radius: int) -> float:
    dist, _ = fastdtw(a, b, radius=radius, dist=lambda x, y: abs(x - y))
    return float(dist)

def retrieve_topk_neighbors(test_seq: Sequence[float], train_bank: List[dict],
                            k: int, radius: int, preselect_k: int = 0) -> List[Tuple[float, dict]]:
    if not train_bank:
        return []
    candidates = train_bank
    if preselect_k and preselect_k > 0 and preselect_k < len(train_bank):
        pre = [(euclidean_sq(test_seq, it["seq"]), it) for it in train_bank]
        candidates = [it for _, it in heapq.nsmallest(preselect_k, pre, key=lambda x: x[0])]
    dists = []
    for it in candidates:
        try:
            d = fastdtw_distance(test_seq, it["seq"], radius=radius)
        except Exception:
            d = float("inf")
        dists.append((d, it))
    return heapq.nsmallest(k, dists, key=lambda x: x[0])

def build_user_prompt_univar_retrieved(classes: List[int], dim_index: int,
                                       test_1d_proc: Sequence[float],
                                       neighbors: List[Tuple[float, dict]],
                                       decimals: int = 3) -> str:
    L = len(test_1d_proc)
    parts = []
    parts.append(f"[Task]\nTime series classification (Channel #{dim_index+1}).")
    parts.append(f"Valid class IDs: {classes}")
    parts.append(f"Input length ≈ {L}. ONLY use this channel; ignore any other channels.")
    parts.append("\n[Retrieved few-shot examples by DTW on this channel]")
    if not neighbors:
        parts.append("  (no neighbors found)")
    else:
        for rank, (dist, it) in enumerate(neighbors, 1):
            seq_str = format_series_1d(it['seq'], decimals)
            parts.append(f"  Example #{rank} (label={it['label']}): {seq_str}")
    parts.append("\n[Unlabeled sample to classify]")
    parts.append(f"Sample: {format_series_1d(test_1d_proc, decimals)}")
    parts.append(
        "\n[Instructions]\n"
        "1. Compare the unlabeled sample ONLY with the retrieved examples shown above.\n"
        "2. Focus on similarity in shape, spikes, oscillations, and recovery patterns. Ignore absolute value scale unless it clearly distinguishes classes.\n"
        "3. If the majority of the retrieved examples have the same label, prioritize that label unless the sample strongly matches another.\n"
        "4. Assign confidence based on neighbor consistency: All neighbors same label ~ 0.9; 2/3 neighbors same label ~ 0.7; Neighbors mixed evenly ~ 0.5\n"
        "5. Return ONLY one JSON object with EXACTLY these keys and no extra text:\n"
        "{\n"
        f'  "decision": <one of {classes}>,\n'
        '  "confidence": <0.0 to 1.0>,\n'
        '  "reasoning": "<one short sentence>"\n'
        "}\n"
    )
    return "\n".join(parts)

FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)

def extract_first_json_object(text: str) -> Optional[str]:
    if not text:
        return None
    s = FENCE_RE.sub("", text).strip()
    start = s.find('{')
    while start != -1:
        i = start; depth = 0; in_str = False; esc = False
        while i < len(s):
            ch = s[i]
            if in_str:
                if esc: esc = False
                elif ch == '\\': esc = True
                elif ch == '"': in_str = False
            else:
                if ch == '"': in_str = True
                elif ch == '{': depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        return s[start:i+1]
            i += 1
        start = s.find('{', start + 1)
    return None

def parse_model_json(s, valid_classes):
    payload = extract_first_json_object(s)
    if not payload:
        return None
    try:
        obj = json.loads(payload)
    except Exception:
        return None
    raw_label = obj.get("decision", obj.get("label"))
    if isinstance(raw_label, str):
        mm = re.search(r"-?\d+", raw_label)
        raw_label = int(mm.group(0)) if mm else None
    elif isinstance(raw_label, (int, float)):
        raw_label = int(raw_label)
    else:
        raw_label = None
    conf = obj.get("confidence", 0.0)
    try:
        conf = float(conf)
    except Exception:
        conf = 0.0
    reason = obj.get("reasoning", obj.get("rationale", ""))
    if raw_label not in valid_classes:
        raw_label = None
    conf = max(0.0, min(1.0, conf))
    return {"label": raw_label, "confidence": conf, "rationale": reason}

def aggregate_conf_weighted(classes: List[int], per_channel: List[dict],
                            eps_other: float = 0.05, clip_lo: float = 0.05, clip_hi: float = 0.95):
    valid = [r for r in per_channel if r.get("label") in classes]
    if not valid:
        return None, 0.0, {"mode": "all_agents_failed"}
    if all(r["label"] == valid[0]["label"] for r in valid):
        c = valid[0]["label"]
        prod = 1.0
        for r in valid:
            conf = max(clip_lo, min(clip_hi, r.get("confidence", 0.0)))
            prod *= (1.0 - conf)
        fused_conf = 1.0 - prod
        return c, min(0.99, fused_conf), {"mode": "consensus"}
    scores = {c: 0.0 for c in classes}
    for r in valid:
        pred = r["label"]
        conf = max(clip_lo, min(clip_hi, r.get("confidence", 0.0)))
        for c in classes:
            scores[c] += conf if c == pred else eps_other
    total = sum(scores.values()) or 1.0
    best_c = max(scores, key=lambda k: scores[k])
    fused_conf = scores[best_c] / total
    return best_c, fused_conf, {"mode": "conf_weighted", "scores": scores}

def run_one_channel(d: int, vals2d: List[List[float]], classes: List[int],
                    per_dim_train: Dict[int, List[dict]], dtw_len: int,
                    z_norm_flag: bool, radius: int, neighbors_k: int,
                    preselect_k: int, decimals: int, model_name: str) -> dict:
    raw_test_1d = vals2d[d]
    test_proc = downsample_1d(raw_test_1d, dtw_len)
    if z_norm_flag:
        test_proc = z_norm_1d(test_proc)

    train_bank = per_dim_train.get(d, [])
    neigh = retrieve_topk_neighbors(
        test_seq=test_proc,
        train_bank=train_bank,
        k=max(1, neighbors_k),
        radius=radius,
        preselect_k=preselect_k
    )

    user_prompt = build_user_prompt_univar_retrieved(
        classes=classes,
        dim_index=d,
        test_1d_proc=test_proc,
        neighbors=neigh,
        decimals=decimals
    )

    ds = get_thread_local_ds(model_name)

    try:
        resp_text = ds.generate(user_prompt)
        parsed = parse_model_json(resp_text, set(classes))
        if parsed is None:
            return {
                "dim": d + 1,
                "label": None,
                "confidence": 0.0,
                "reasoning": "parse_error",
                "neighbors": [
                    {"train_id": it["id"], "label": it["label"], "dtw_dist": float(dist)}
                    for dist, it in neigh
                ],
                "raw": resp_text
            }
        return {
            "dim": d + 1,
            "label": parsed["label"],
            "confidence": parsed["confidence"],
            "reasoning": parsed["rationale"],
            "neighbors": [
                {"train_id": it["id"], "label": it["label"], "dtw_dist": float(dist)}
                for dist, it in neigh
            ],
            "raw": resp_text
        }
    except Exception as e:
        return {
            "dim": d + 1,
            "label": None,
            "confidence": 0.0,
            "reasoning": f"exception: {e}",
            "neighbors": [
                {"train_id": it["id"], "label": it["label"], "dtw_dist": float(dist)}
                for dist, it in neigh
            ],
            "raw": ""
        }

def main():
    args = parse_arguments()
    random.seed(args.seed)

    train = load_json(args.train_json)
    test = load_json(args.test_json)
    classes = get_classes_from_train(train)

    dtw_len = args.dtw_len if args.dtw_len is not None else args.max_len
    radius = args.dtw_radius if args.dtw_radius is not None else max(1, int(round(args.dtw_radius_frac * max(2, dtw_len))))
    z_norm_flag = bool(args.z_norm)
    per_dim_train = build_per_dim_train_cache(train, dtw_len=dtw_len, z_norm=z_norm_flag)

    outputs = []
    correct = 0
    agent_calls = 0
    agent_bad_json = 0
    agent_bad_label = 0
    samples_all_failed = 0
    workers = max(1, int(args.workers))
    executor = ThreadPoolExecutor(max_workers=workers)

    pbar = tqdm(total=len(test), desc="Classifying (per-sample concurrent agents)", unit="sample", dynamic_ncols=True)
    for i, row in enumerate(test):
        gold_label = row.get("label")
        vals2d = ensure_2d(row.get("values") or [])
        D = len(vals2d)
        use_dims = D if args.max_dims is None else min(D, max(1, args.max_dims))
        dims_to_use = list(range(use_dims))
        futures = {}
        for d in dims_to_use:
            fut = executor.submit(
                run_one_channel, d, vals2d, classes, per_dim_train,
                dtw_len, z_norm_flag, radius, args.neighbors_k,
                args.preselect_k, args.decimals, args.model_name
            )
            futures[fut] = d

        per_channel = []
        for fut in as_completed(futures):
            d = futures[fut]
            res = fut.result()
            per_channel.append(res)

        agent_calls += len(dims_to_use)
        for r in per_channel:
            if r.get("label") is None:
                agent_bad_json += 1
            elif r.get("label") not in classes:
                agent_bad_label += 1

        final_label, final_conf, agg_info = aggregate_conf_weighted(
            classes=classes,
            per_channel=per_channel,
            eps_other=args.eps_other,
            clip_lo=args.clip_lo,
            clip_hi=args.clip_hi
        )
        if final_label is None:
            samples_all_failed += 1

        is_correct = (final_label == gold_label)
        correct += int(is_correct)

        outputs.append({
            "id": row.get("id", i),
            "gold": gold_label,
            "pred": final_label,
            "confidence": final_conf,
            "aggregation": agg_info,
            "per_channel": sorted(per_channel, key=lambda x: x["dim"])
        })

        pbar.update(1)
        running_acc = correct / (i + 1)
        pbar.set_postfix(acc=f"{running_acc:.4f}")

    pbar.close()
    executor.shutdown(wait=True)

    result = {
        "meta": {
            "provider": "deepseek",
            "model": args.model_name,
            "max_len": args.max_len,
            "decimals": args.decimals,
            "max_dims": args.max_dims,
            "num_agents_per_sample": "all" if args.max_dims is None else args.max_dims,
            "aggregation": "conf_weighted_clip_eps",
            "eps_other": args.eps_other,
            "clip": [args.clip_lo, args.clip_hi],
            "classes": classes,
            "retrieval": {
                "mode": "fastdtw_topK",
                "neighbors_k": args.neighbors_k,
                "dtw_len": dtw_len,
                "dtw_radius": radius,
                "dtw_radius_frac": None if args.dtw_radius is not None else args.dtw_radius_frac,
                "z_norm": z_norm_flag,
                "preselect_k": args.preselect_k
            },
            "concurrency": {"workers": workers, "thread_local_client": True}
        },
        "accuracy": correct / max(1, len(test)),
        "stats": {
            "num_samples": len(test),
            "agent_calls": agent_calls,
            "agent_bad_json": agent_bad_json,
            "agent_bad_label": agent_bad_label,
            "samples_all_agents_failed": samples_all_failed
        },
        "predictions": outputs
    }

    out_dir = os.path.dirname(args.output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"[done] saved to {args.output_file}. "
          f"final acc={result['accuracy']:.4f} | "
          f"agent_calls={agent_calls}, bad_json={agent_bad_json}, bad_label={agent_bad_label}, "
          f"samples_all_agents_failed={samples_all_failed} | workers={workers}")

if __name__ == "__main__":
    main()
