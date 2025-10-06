import json
import os
import math
import argparse
import random
from collections import defaultdict
from typing import Any, List, Sequence, Dict, Tuple, Optional

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
    return [float(values[i]) for i in idxs]

def z_norm_1d(values: Sequence[float]) -> List[float]:
    n = len(values)
    if n == 0:
        return []
    mu = sum(values) / n
    var = sum((v - mu) * (v - mu) for v in values) / n
    if var <= 1e-12:
        return [float(v - mu) for v in values]
    sd = math.sqrt(var)
    return [float((v - mu) / sd) for v in values]

def euclidean_sq(a: Sequence[float], b: Sequence[float]) -> float:
    return sum((x - y) * (x - y) for x, y in zip(a, b))

def l2norm(x: Sequence[float]) -> float:
    return math.sqrt(sum(v*v for v in x))

def cosine_sim(a: Sequence[float], b: Sequence[float]) -> float:
    na = l2norm(a); nb = l2norm(b)
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return sum(x*y for x, y in zip(a, b)) / (na * nb)

def zscore_list(xs: List[float]) -> List[float]:
    if not xs:
        return []
    mu = sum(xs) / len(xs)
    var = sum((x - mu) ** 2 for x in xs) / max(1, len(xs) - 1)
    sd = math.sqrt(var) if var > 0 else 1.0
    return [(x - mu) / sd for x in xs]

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def preprocess_train_per_channel(train: List[dict], max_len: int, z_norm: bool) -> Tuple[Dict[int, List[List[float]]], List[int]]:
    min_dims = None
    for r in train:
        vals2d = ensure_2d(r.get("values"))
        d = len(vals2d)
        min_dims = d if (min_dims is None or d < min_dims) else min_dims
    if min_dims is None:
        raise ValueError("Empty training data.")

    labels = [int(r["label"]) for r in train]
    per_dim_values: Dict[int, List[List[float]]] = {d: [] for d in range(min_dims)}

    for r in train:
        vals2d = ensure_2d(r.get("values"))[:min_dims]
        for d, seq in enumerate(vals2d):
            x = downsample_1d(seq, max_len)
            if z_norm:
                x = z_norm_1d(x)
            per_dim_values[d].append(x)

    return per_dim_values, labels

def score_prototype_margin(channel_data: List[List[float]], labels: List[int]) -> Tuple[float, Dict]:
    by_cls: Dict[int, List[List[float]]] = defaultdict(list)
    for x, y in zip(channel_data, labels):
        by_cls[int(y)].append(x)
    classes = sorted(by_cls.keys())
    if len(classes) <= 1:
        return 0.0, {"note": "single_class", "classes": classes}

    centroids: Dict[int, List[float]] = {}
    for c, xs in by_cls.items():
        L = len(xs[0]) if xs else 0
        if L == 0:
            centroids[c] = []
            continue
        mu = [0.0] * L
        for v in xs:
            for i in range(L):
                mu[i] += v[i]
        n = max(1, len(xs))
        mu = [vv / n for vv in mu]
        centroids[c] = mu

    spreads = []
    for c, xs in by_cls.items():
        mu = centroids[c]
        if not xs or not mu:
            continue
        dsum = 0.0
        for v in xs:
            dsum += math.sqrt(euclidean_sq(v, mu))
        spreads.append(dsum / max(1, len(xs)))
    mean_within = sum(spreads) / max(1, len(spreads))

    pair_dists = []
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            ci, cj = classes[i], classes[j]
            ai, aj = centroids[ci], centroids[cj]
            if ai and aj:
                pair_dists.append(math.sqrt(euclidean_sq(ai, aj)))
    mean_between = sum(pair_dists) / max(1, len(pair_dists))

    eps = 1e-8
    score = mean_between / (mean_within + eps)
    dbg = {
        "classes": classes,
        "mean_between": mean_between,
        "mean_within": mean_within,
        "num_pairs": len(pair_dists),
    }
    return score, dbg

def score_1nn_accuracy(channel_data: List[List[float]], labels: List[int], nn_eval_samples: int, seed: int = 42) -> Tuple[float, Dict]:
    N = len(channel_data)
    if N <= 1:
        return 0.0, {"note": "too_few_samples"}

    idxs = list(range(N))
    rnd = random.Random(seed)
    if nn_eval_samples <= 0 or nn_eval_samples >= N:
        probe = idxs
    else:
        probe = rnd.sample(idxs, nn_eval_samples)

    correct = 0
    for i in probe:
        xi = channel_data[i]
        yi = labels[i]
        best_d, best_j = None, None
        for j in idxs:
            if j == i:
                continue
            d = euclidean_sq(xi, channel_data[j])
            if (best_d is None) or (d < best_d):
                best_d, best_j = d, j
        if best_j is not None and labels[best_j] == yi:
            correct += 1

    acc = correct / max(1, len(probe))
    return acc, {"N_probe": len(probe)}

def build_centroid_stack(channel_data: List[List[float]], labels: List[int]) -> List[float]:
    by_cls: Dict[int, List[List[float]]] = defaultdict(list)
    for x, y in zip(channel_data, labels):
        by_cls[int(y)].append(x)
    classes = sorted(by_cls.keys())
    stacks: List[float] = []
    for c in classes:
        xs = by_cls[c]
        if not xs:
            continue
        L = len(xs[0])
        mu = [0.0] * L
        for v in xs:
            for i in range(L):
                mu[i] += v[i]
        n = max(1, len(xs))
        mu = [vv / n for vv in mu]
        stacks.extend(mu)
    return stacks

def parse_args():
    ap = argparse.ArgumentParser(description="Channel selection by prototype/1NN margins")
    ap.add_argument("--train_json", type=str, required=True, help="Path to TRAIN json")
    ap.add_argument("--output_json", type=str, required=True, help="Where to save selected indices & scores")
    ap.add_argument("--max_len", type=int, default=384, help="Downsample length for each 1D channel series")
    ap.add_argument("--z_norm", type=int, default=0, help="1: z-score per sample; 0: off")
    ap.add_argument("--budget_m", type=int, default=10, help="Keep top-M channels by fused score")
    ap.add_argument("--alpha", type=float, default=0.7, help="Fusion weight for B (prototype margin); (1-alpha) for C (1NN acc)")
    ap.add_argument("--nn_eval_samples", type=int, default=200, help="Probe size for fast 1NN LOO (<=0 means use all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--diversity_threshold", type=float, default=None,
                    help="If set (e.g., 0.98), prune channels whose centroid-stack cosine similarity with a selected channel exceeds this threshold")

    return ap.parse_args()

def main():
    args = parse_args()
    random.seed(args.seed)

    train = load_json(args.train_json)
    per_dim_values, labels = preprocess_train_per_channel(
        train=train,
        max_len=args.max_len,
        z_norm=bool(args.z_norm),
    )
    dims = sorted(per_dim_values.keys())

    scores_B, scores_C, debug_B, debug_C = [], [], {}, {}
    centroid_stacks: Dict[int, List[float]] = {}

    for d in dims:
        data_d = per_dim_values[d]

        b, dbgB = score_prototype_margin(data_d, labels)
        scores_B.append(b)
        debug_B[d] = dbgB

        c, dbgC = score_1nn_accuracy(data_d, labels, nn_eval_samples=args.nn_eval_samples, seed=args.seed + d)
        scores_C.append(c)
        debug_C[d] = dbgC

        centroid_stacks[d] = build_centroid_stack(data_d, labels)

    zb = zscore_list(scores_B)
    zc = zscore_list(scores_C)

    fused = []
    for i, d in enumerate(dims):
        s = args.alpha * zb[i] + (1 - args.alpha) * zc[i]
        fused.append((d, s))

    fused.sort(key=lambda x: x[1], reverse=True)

    selected = []
    if args.diversity_threshold is None or args.diversity_threshold <= 0:
        selected = [d for d, _ in fused[:args.budget_m]]
    else:
        for d, _ in fused:
            vec_d = centroid_stacks[d]
            ok = True
            for k in selected:
                sim = cosine_sim(vec_d, centroid_stacks[k])
                if sim > args.diversity_threshold:
                    ok = False
                    break
            if ok:
                selected.append(d)
            if len(selected) >= args.budget_m:
                break

    out = {
        "config": {
            "max_len": args.max_len,
            "z_norm": bool(args.z_norm),
            "budget_m": args.budget_m,
            "alpha": args.alpha,
            "nn_eval_samples": args.nn_eval_samples,
            "diversity_threshold": args.diversity_threshold,
            "seed": args.seed,
        },
        "selected_indices": selected,
        "scores": {
            "per_channel": [
                {
                    "dim": d,
                    "score_fused": float(next(s for dd, s in fused if dd == d)),
                    "score_B_prototype_margin": float(scores_B[i]),
                    "score_C_1nn_acc": float(scores_C[i]),
                    "debug_B": debug_B[d],
                    "debug_C": debug_C[d],
                }
                for i, d in enumerate(dims)
            ],
            "ranked_dims_desc": [d for d, _ in fused],
        }
    }

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"[done] saved to {args.output_json}")
    print(f"Top-{args.budget_m} channel indices:", selected)

if __name__ == "__main__":
    main()
