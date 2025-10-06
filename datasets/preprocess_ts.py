import json
import argparse
import math
from typing import List, Optional, Dict, Tuple

def _parse_ts_lines(ts_filename: str):
    in_data = False
    with open(ts_filename, 'r', encoding='utf-8') as f:
        for idx, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            if line.startswith('@'):
                if line.lower().startswith('@data'):
                    in_data = True
                continue
            if not in_data:
                continue
            yield idx, line

def _build_label_map_from_file(ts_filename: str) -> Dict[str, int]:
    labels_in_order: List[str] = []
    seen = set()
    for _, line in _parse_ts_lines(ts_filename):
        parts = [p.strip() for p in line.split(':')]
        if len(parts) < 2:
            continue
        lab = parts[-1]
        if lab not in seen:
            seen.add(lab)
            labels_in_order.append(lab)
    return {lab: i for i, lab in enumerate(labels_in_order)}

def _ensure_label_map(label_map_or_path: Optional[str], ts_filename: str) -> Dict[str, int]:
    if isinstance(label_map_or_path, str):
        with open(label_map_or_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return _build_label_map_from_file(ts_filename)

def _is_finite_number_token(tok: str) -> bool:
    tl = tok.lower()
    if tl in ("nan", "inf", "-inf", "+inf"):
        return False
    try:
        _ = float(tok)
        return math.isfinite(float(tok))
    except Exception:
        return False

def process_ts_to_json(
    ts_filename: str,
    json_filename: str,
    keep_dims: Optional[List[int]] = None,
    decimals: int = 5,
    label_map: Optional[str] = None,
    save_label_map: bool = True,
    label_map_out: Optional[str] = None,
    allow_new_labels: bool = False
) -> Tuple[List[dict], Dict[str, int]]:
    label2id = _ensure_label_map(label_map, ts_filename)

    data = []
    for idx, line in _parse_ts_lines(ts_filename):
        if ':' not in line:
            raise ValueError(f"[{ts_filename}] Line {idx} has no ':' separator: {line[:120]}")
        parts = [p.strip() for p in line.split(':')]
        if len(parts) < 2:
            raise ValueError(f"[{ts_filename}] Line {idx} malformed: {line[:120]}")

        label_raw = parts[-1]
        dim_parts = parts[:-1]

        if keep_dims is None:
            sel_dim_idxs = list(range(len(dim_parts)))
        else:
            sel_dim_idxs = []
            for k in keep_dims:
                if k < 1 or k > len(dim_parts):
                    raise ValueError(f"[{ts_filename}] Line {idx}: keep_dims includes invalid dim {k}, "
                                     f"but this sample has {len(dim_parts)} dims")
                sel_dim_idxs.append(k - 1)

        values_2d = []
        length_ref = None
        for d0 in sel_dim_idxs:
            dim_str = dim_parts[d0]
            tokens = [t.strip() for t in dim_str.split(',') if t.strip()]
            for t in tokens:
                if not _is_finite_number_token(t):
                    raise ValueError(f"[{ts_filename}] Line {idx}: invalid token '{t}' in dim#{d0+1}")

            try:
                dim_vals = [round(float(t), decimals) for t in tokens]
            except ValueError as e:
                raise ValueError(f"[{ts_filename}] Line {idx}: non-numeric token in dim#{d0+1}: {e}")

            if length_ref is None:
                length_ref = len(dim_vals)
            elif len(dim_vals) != length_ref:
                raise ValueError(
                    f"[{ts_filename}] Line {idx}: per-dimension lengths differ, "
                    f"dim1={length_ref}, dim{d0+1}={len(dim_vals)}"
                )
            values_2d.append(dim_vals)

        if label_raw not in label2id:
            if allow_new_labels:
                next_id = (max(label2id.values()) + 1) if label2id else 0
                label2id[label_raw] = next_id
            else:
                raise ValueError(
                    f"[{ts_filename}] Unknown label '{label_raw}' not present in provided label_map. "
                    f"Pass TRAIN's label_map to TEST, or use --allow-new-labels if you really want to extend."
                )
        label_id = label2id[label_raw]

        data.append({
            "id": len(data),
            "label": label_id,
            "label_raw": label_raw,
            "values": values_2d,
            "num_dims": len(values_2d),
            "length": length_ref or 0
        })

    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    if save_label_map:
        map_path = label_map_out or json_filename.replace(".json", "_labelmap.json")
        with open(map_path, 'w', encoding='utf-8') as f:
            json.dump(label2id, f, indent=2, ensure_ascii=False)

    print(f"[done] {ts_filename} -> {json_filename} "
          f"({len(data)} samples, dims={data[0]['num_dims'] if data else 0}, len={data[0]['length'] if data else 0}).")
    return data, label2id

def _parse_keep_dims(arg: Optional[str]) -> Optional[List[int]]:
    if not arg:
        return None
    out: List[int] = []
    for tok in arg.replace(',', ' ').split():
        out.append(int(tok))
    return out or None

def main():
    ap = argparse.ArgumentParser(description="Preprocess UCR/UEA .ts into JSON (stable label mapping, robust parsing).")
    ap.add_argument("--ts", required=True, help="Path to .ts file")
    ap.add_argument("--json", required=True, help="Output JSON path")
    ap.add_argument("--keep-dims", type=str, default=None,
                    help="1-based dim indices to keep, e.g. '1,2,3'. Default: keep all")
    ap.add_argument("--decimals", type=int, default=5, help="Round floats to this many decimals")
    ap.add_argument("--label-map", type=str, default=None,
                    help="Path to existing TRAIN label_map JSON (for TEST). If omitted, a new map is built from this file.")
    ap.add_argument("--save-label-map", action="store_true",
                    help="Save (or update) label map JSON alongside the output JSON (default name: *_labelmap.json).")
    ap.add_argument("--label-map-out", type=str, default=None,
                    help="Custom path to write label map JSON (takes effect only with --save-label-map).")
    ap.add_argument("--allow-new-labels", action="store_true",
                    help="Allow extending label map when encountering unseen labels (NOT recommended for TEST).")
    args = ap.parse_args()

    keep_dims = _parse_keep_dims(args.keep_dims)
    process_ts_to_json(
        ts_filename=args.ts,
        json_filename=args.json,
        keep_dims=keep_dims,
        decimals=args.decimals,
        label_map=args.label_map,
        save_label_map=bool(args.save_label_map),
        label_map_out=args.label_map_out,
        allow_new_labels=bool(args.allow_new_labels),
    )

if __name__ == "__main__":
    main()
