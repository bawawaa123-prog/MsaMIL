#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

LABEL_TO_INT = {
    "NonAdenocarcinoma": 0,
    "Adenocarcinoma": 1,
}


def _read_prob_map(pred_csv: Path) -> Dict[str, float]:
    if not pred_csv.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_csv}")

    out: Dict[str, float] = {}
    with pred_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"wsi_id", "prob_Adenocarcinoma"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns in {pred_csv}: {sorted(missing)}")

        for row in reader:
            sid = str(row.get("wsi_id", "")).strip()
            prob_str = str(row.get("prob_Adenocarcinoma", "")).strip()
            if not sid or not prob_str:
                continue
            try:
                prob = float(prob_str)
            except ValueError:
                continue
            if sid not in out:
                out[sid] = prob
    return out


def _merge_prob_maps(maps: List[Dict[str, float]]) -> Tuple[Dict[str, float], List[str]]:
    merged: Dict[str, float] = {}
    dup_ids: List[str] = []
    for m in maps:
        for sid, prob in m.items():
            if sid in merged:
                dup_ids.append(sid)
                continue
            merged[sid] = prob
    return merged, dup_ids


def _build_rows(all_data_csv: Path, prob_map: Dict[str, float], keep_missing: bool) -> Tuple[List[List[object]], List[str]]:
    if not all_data_csv.exists():
        raise FileNotFoundError(f"all_data.csv not found: {all_data_csv}")

    rows: List[List[object]] = []
    missing_ids: List[str] = []

    with all_data_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"slide_id", "label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns in {all_data_csv}: {sorted(missing)}")

        for row in reader:
            sid = str(row["slide_id"]).strip()
            label_name = str(row["label"]).strip()
            if label_name not in LABEL_TO_INT:
                raise ValueError(
                    f"Unknown label '{label_name}' for slide '{sid}'. "
                    f"Expected one of {sorted(LABEL_TO_INT)}"
                )
            true_label = LABEL_TO_INT[label_name]

            if sid in prob_map:
                rows.append([sid, true_label, prob_map[sid]])
            else:
                missing_ids.append(sid)
                if keep_missing:
                    rows.append([sid, true_label, ""])

    return rows, missing_ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge train/val/test prediction CSVs and export p-value input CSV in all_data order."
    )
    parser.add_argument(
        "--all-data-csv",
        type=Path,
        default=Path("MsaMIL_Net/data/all_data.csv"),
        help="Path to all_data.csv",
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=Path("MsaMIL_Net/results/YiYuan/features_phikon_queries_10/per_wsi_predictions/epoch_026_train_predictions.csv"),
        help="Train predictions CSV",
    )
    parser.add_argument(
        "--val-csv",
        type=Path,
        default=Path("MsaMIL_Net/results/YiYuan/features_phikon_queries_10/per_wsi_predictions/epoch_026_val_predictions.csv"),
        help="Val predictions CSV",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=Path("MsaMIL_Net/results/YiYuan/features_phikon_queries_10/per_wsi_predictions/test_predictions_best_20260107_115422.csv"),
        help="Test predictions CSV",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("MsaMIL_Net/data/pvalue_input_train_val_test.csv"),
        help="Output CSV path in data directory",
    )
    parser.add_argument(
        "--missing-report",
        type=Path,
        default=Path("MsaMIL_Net/data/pvalue_input_train_val_test_missing_slides.txt"),
        help="File to save missing slide IDs",
    )
    parser.add_argument(
        "--keep-missing",
        action="store_true",
        help="Keep rows even when probability is missing (empty probability field)",
    )
    args = parser.parse_args()

    train_map = _read_prob_map(args.train_csv)
    val_map = _read_prob_map(args.val_csv)
    test_map = _read_prob_map(args.test_csv)

    merged_map, dup_ids = _merge_prob_maps([train_map, val_map, test_map])

    rows, missing_ids = _build_rows(
        all_data_csv=args.all_data_csv,
        prob_map=merged_map,
        keep_missing=bool(args.keep_missing),
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["slide_id", "true_label", "prob_Adenocarcinoma"])
        writer.writerows(rows)

    if missing_ids:
        args.missing_report.parent.mkdir(parents=True, exist_ok=True)
        with args.missing_report.open("w", encoding="utf-8") as f:
            for sid in missing_ids:
                f.write(f"{sid}\n")

    print(f"[Done] output_csv: {args.output_csv}")
    print(
        f"[Stats] train={len(train_map)}, val={len(val_map)}, test={len(test_map)}, "
        f"merged={len(merged_map)}, rows={len(rows)}, missing={len(missing_ids)}, duplicates={len(dup_ids)}"
    )
    if dup_ids:
        print("[Warn] Duplicate slide IDs across split csvs were detected; first occurrence was kept.")
    if missing_ids:
        print(f"[Done] missing_report: {args.missing_report}")


if __name__ == "__main__":
    main()
