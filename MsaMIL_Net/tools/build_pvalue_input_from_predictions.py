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


def _read_prediction_prob_map(csv_path: Path) -> Dict[str, float]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {csv_path}")

    out: Dict[str, float] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"wsi_id", "prob_Adenocarcinoma"}
        missing_cols = required - set(reader.fieldnames or [])
        if missing_cols:
            raise ValueError(f"Missing columns in {csv_path}: {sorted(missing_cols)}")

        for row in reader:
            wsi_id = str(row["wsi_id"]).strip()
            prob_str = str(row["prob_Adenocarcinoma"]).strip()
            if not wsi_id or not prob_str:
                continue
            try:
                prob = float(prob_str)
            except ValueError:
                continue
            if wsi_id not in out:
                out[wsi_id] = prob
    return out


def _build_rows(
    all_data_csv: Path,
    pred_prob_map: Dict[str, float],
    *,
    keep_missing: bool,
) -> Tuple[List[List[object]], List[str]]:
    if not all_data_csv.exists():
        raise FileNotFoundError(f"all_data CSV not found: {all_data_csv}")

    rows_out: List[List[object]] = []
    missing_slides: List[str] = []

    with all_data_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"slide_id", "label"}
        missing_cols = required - set(reader.fieldnames or [])
        if missing_cols:
            raise ValueError(f"Missing columns in {all_data_csv}: {sorted(missing_cols)}")

        for row in reader:
            slide_id = str(row["slide_id"]).strip()
            label_name = str(row["label"]).strip()

            if label_name not in LABEL_TO_INT:
                raise ValueError(
                    f"Unknown label '{label_name}' for slide '{slide_id}'. "
                    f"Expected one of: {sorted(LABEL_TO_INT)}"
                )
            true_label = LABEL_TO_INT[label_name]

            if slide_id in pred_prob_map:
                prob = pred_prob_map[slide_id]
                rows_out.append([slide_id, true_label, prob])
            else:
                missing_slides.append(slide_id)
                if keep_missing:
                    rows_out.append([slide_id, true_label, ""])  # empty probability

    return rows_out, missing_slides


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build p-value input CSV with columns: slide_id, true_label, prob_Adenocarcinoma. "
            "slide order follows all_data.csv."
        )
    )
    parser.add_argument(
        "--all-data-csv",
        type=Path,
        default=Path("MsaMIL_Net/data/all_data.csv"),
        help="Path to all_data.csv containing slide_id and label",
    )
    parser.add_argument(
        "--train-pred-csv",
        type=Path,
        default=Path("MsaMIL_Net/results/YiYuan/features_phikon_queries_10/per_wsi_predictions/epoch_026_train_predictions.csv"),
        help="Path to train predictions CSV",
    )
    parser.add_argument(
        "--val-pred-csv",
        type=Path,
        default=Path("MsaMIL_Net/results/YiYuan/features_phikon_queries_10/per_wsi_predictions/epoch_026_val_predictions.csv"),
        help="Path to val predictions CSV",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("MsaMIL_Net/results/YiYuan/features_phikon_queries_10/per_wsi_predictions/epoch_026_pvalue_input.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--keep-missing",
        action="store_true",
        help="Keep missing slides with empty probability instead of dropping",
    )
    parser.add_argument(
        "--missing-report",
        type=Path,
        default=Path("MsaMIL_Net/results/YiYuan/features_phikon_queries_10/per_wsi_predictions/epoch_026_pvalue_missing_slides.txt"),
        help="Path to save missing slide IDs report",
    )
    args = parser.parse_args()

    train_map = _read_prediction_prob_map(args.train_pred_csv)
    val_map = _read_prediction_prob_map(args.val_pred_csv)

    pred_prob_map: Dict[str, float] = {}
    dup_count = 0

    for k, v in train_map.items():
        pred_prob_map[k] = v

    for k, v in val_map.items():
        if k in pred_prob_map:
            dup_count += 1
            # Keep train value by default; this should rarely happen for disjoint train/val splits.
            continue
        pred_prob_map[k] = v

    rows, missing = _build_rows(
        args.all_data_csv,
        pred_prob_map,
        keep_missing=bool(args.keep_missing),
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["slide_id", "true_label", "prob_Adenocarcinoma"])
        writer.writerows(rows)

    if missing:
        args.missing_report.parent.mkdir(parents=True, exist_ok=True)
        with args.missing_report.open("w", encoding="utf-8") as f:
            for sid in missing:
                f.write(f"{sid}\n")

    print(f"[Done] output: {args.output_csv}")
    print(f"[Stats] train={len(train_map)}, val={len(val_map)}, merged={len(pred_prob_map)}")
    print(f"[Stats] rows_written={len(rows)}, missing={len(missing)}, duplicates_train_val={dup_count}")
    if missing:
        print(f"[Done] missing report: {args.missing_report}")


if __name__ == "__main__":
    main()
