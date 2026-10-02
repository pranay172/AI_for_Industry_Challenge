#!/usr/bin/env python3
"""Offline inference/evaluation for the SC port heatmap detector."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the aic_model package

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from aic_model.vision_runtime import transforms

from train_sc_port_detector import (
    LANDMARK_NAMES,
    ScPortHeatmapNet,
    heatmap_argmax,
    load_manifest,
)


@dataclass
class EvalRow:
    sample_id: str
    phase: str
    camera: str
    target_module_name: str
    port_name: str
    landmark: str
    visible_gt: int
    visible_pred: int
    confidence: float
    gt_x_px: float
    gt_y_px: float
    pred_x_px: float
    pred_y_px: float
    err_px: float
    err_model_px: float


def load_checkpoint(path: Path, device: torch.device):
    ckpt = torch.load(path.expanduser(), map_location=device, weights_only=True)
    if ckpt.get("preprocessing", "full_frame_v1") != "full_frame_v1":
        raise ValueError("This evaluator measures full-frame inputs. Use the trainer crop validation for crop checkpoints; runtime rail-crop evaluation is a separate requirement.")
    names = tuple(ckpt.get("landmark_names", LANDMARK_NAMES))
    if names != LANDMARK_NAMES:
        raise SystemExit(
            f"Checkpoint landmarks {names} do not match expected {LANDMARK_NAMES}"
        )
    model = ScPortHeatmapNet(num_landmarks=len(LANDMARK_NAMES)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    model.decoder = ckpt.get("decoder", "argmax_v1")
    if model.decoder not in {"argmax_v1", "local_log_quadratic_v1"}:
        raise ValueError(f"Unsupported decoder: {model.decoder}")
    img_size = int(ckpt.get("img_size", 224))
    heatmap_size = int(ckpt.get("heatmap_size", img_size // 4))
    return model, img_size, heatmap_size, ckpt


def load_image(sample) -> Image.Image:
    npz = np.load(sample.npz_path)
    return Image.fromarray(npz[sample.image_key]).convert("RGB")


def image_tensor(img: Image.Image, img_size: int) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return transform(img).unsqueeze(0)


def infer_sample(
    model,
    sample,
    img: Image.Image,
    img_size: int,
    heatmap_size: int,
    device: torch.device,
):
    with torch.no_grad():
        pred_heatmaps, pred_vis = model(image_tensor(img, img_size).to(device))
        heatmaps_prob = torch.sigmoid(pred_heatmaps)
        pred_pts_hm = heatmap_argmax(heatmaps_prob, refine=model.decoder == "local_log_quadratic_v1")[0].cpu().numpy()
        vis_prob = torch.sigmoid(pred_vis)[0].cpu().numpy()
        heatmap_conf = (
            heatmaps_prob.reshape(1, len(LANDMARK_NAMES), -1)
            .max(dim=-1)[0][0]
            .cpu()
            .numpy()
        )

    sx = img.width / float(heatmap_size)
    sy = img.height / float(heatmap_size)
    pred_px = np.column_stack([pred_pts_hm[:, 0] * sx, pred_pts_hm[:, 1] * sy])
    gt_norm = np.array(sample.points_norm, dtype=np.float64)
    gt_px = np.column_stack([gt_norm[:, 0] * img.width, gt_norm[:, 1] * img.height])
    pred_model_px = pred_pts_hm * (img_size / float(heatmap_size))
    gt_model_px = gt_norm * img_size
    visible = np.array(sample.visible, dtype=bool)
    confidence = np.sqrt(np.clip(vis_prob, 0.0, 1.0) * np.clip(heatmap_conf, 0.0, 1.0))
    return gt_px, pred_px, gt_model_px, pred_model_px, visible, confidence, vis_prob


def summarize_rows(rows: list[EvalRow]) -> list[dict]:
    groups: dict[tuple[str, str], list[EvalRow]] = {}
    for row in rows:
        groups.setdefault(("all", "all"), []).append(row)
        groups.setdefault(("phase", row.phase), []).append(row)
        groups.setdefault(("camera", row.camera), []).append(row)
        groups.setdefault(("landmark", row.landmark), []).append(row)
        groups.setdefault(("target_module", row.target_module_name), []).append(row)

    out = []
    for (group_type, group_id), group_rows in sorted(groups.items()):
        errs = sorted(r.err_px for r in group_rows if math.isfinite(r.err_px))
        model_errs = sorted(
            r.err_model_px for r in group_rows if math.isfinite(r.err_model_px)
        )

        def _stats(values: list[float]) -> tuple[float, float, float, float]:
            if not values:
                return float("nan"), float("nan"), float("nan"), float("nan")
            return (
                sum(values) / len(values),
                values[int(0.5 * (len(values) - 1))],
                values[int(0.9 * (len(values) - 1))],
                values[-1],
            )

        mean, median, p90, max_err = _stats(errs)
        model_mean, model_median, model_p90, model_max = _stats(model_errs)
        vis_acc = (
            100.0
            * sum(r.visible_gt == r.visible_pred for r in group_rows)
            / max(len(group_rows), 1)
        )
        out.append(
            {
                "group_type": group_type,
                "group_id": group_id,
                "n": len(group_rows),
                "err_mean_px": mean,
                "err_median_px": median,
                "err_p90_px": p90,
                "err_max_px": max_err,
                "err_model_mean_px": model_mean,
                "err_model_median_px": model_median,
                "err_model_p90_px": model_p90,
                "err_model_max_px": model_max,
                "visibility_acc": vis_acc,
            }
        )
    return out


def write_rows_csv(path: Path, rows: list[EvalRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(EvalRow.__dataclass_fields__.keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: getattr(row, name) for name in fieldnames})


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "group_type",
        "group_id",
        "n",
        "err_mean_px",
        "err_median_px",
        "err_p90_px",
        "err_max_px",
        "err_model_mean_px",
        "err_model_median_px",
        "err_model_p90_px",
        "err_model_max_px",
        "visibility_acc",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def draw_overlay(
    img: Image.Image,
    sample,
    gt_px: np.ndarray,
    pred_px: np.ndarray,
    confidence: np.ndarray,
    output_path: Path,
) -> None:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    colors = [
        (255, 80, 80),
        (255, 180, 80),
        (80, 220, 120),
        (80, 160, 255),
        (255, 80, 220),
    ]
    for idx, name in enumerate(LANDMARK_NAMES):
        color = colors[idx]
        gx, gy = gt_px[idx]
        px, py = pred_px[idx]
        if sample.visible[idx] and gx >= 0.0 and gy >= 0.0:
            draw.ellipse([gx - 4, gy - 4, gx + 4, gy + 4], outline=color, width=2)
        draw.line([px - 5, py, px + 5, py], fill=color, width=2)
        draw.line([px, py - 5, px, py + 5], fill=color, width=2)
        draw.text((px + 6, py - 5), f"{name}:{confidence[idx]:.2f}", fill=color)

    if all(sample.visible[:4]):
        face_gt = [tuple(gt_px[i]) for i in range(4)]
        face_pred = [tuple(pred_px[i]) for i in range(4)]
        draw.line(face_gt + [face_gt[0]], fill=(255, 255, 255), width=1)
        draw.line(face_pred + [face_pred[0]], fill=(0, 0, 0), width=2)

    draw.rectangle([0, 0, out.width, 20], fill=(0, 0, 0))
    draw.text(
        (4, 4),
        f"{sample.sample_id} {sample.phase} {sample.camera} {sample.target_module_name}",
        fill=(255, 255, 255),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(output_path)


def evaluate(args) -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    model, img_size, heatmap_size, ckpt = load_checkpoint(args.model, device)
    samples = load_manifest(args.labels.expanduser())
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
    if not samples:
        raise SystemExit(f"No usable rows in {args.labels}")

    rows: list[EvalRow] = []
    overlays_saved = 0
    for sample in samples:
        img = load_image(sample)
        (
            gt_px,
            pred_px,
            gt_model_px,
            pred_model_px,
            visible,
            confidence,
            vis_prob,
        ) = infer_sample(
            model,
            sample,
            img,
            img_size,
            heatmap_size,
            device,
        )
        for idx, name in enumerate(LANDMARK_NAMES):
            err = (
                float(np.linalg.norm(pred_px[idx] - gt_px[idx]))
                if visible[idx]
                else float("nan")
            )
            err_model = (
                float(np.linalg.norm(pred_model_px[idx] - gt_model_px[idx]))
                if visible[idx]
                else float("nan")
            )
            rows.append(
                EvalRow(
                    sample_id=sample.sample_id,
                    phase=sample.phase,
                    camera=sample.camera,
                    target_module_name=sample.target_module_name,
                    port_name=sample.port_name,
                    landmark=name,
                    visible_gt=int(visible[idx]),
                    visible_pred=int(vis_prob[idx] >= args.visibility_threshold),
                    confidence=float(confidence[idx]),
                    gt_x_px=float(gt_px[idx, 0]),
                    gt_y_px=float(gt_px[idx, 1]),
                    pred_x_px=float(pred_px[idx, 0]),
                    pred_y_px=float(pred_px[idx, 1]),
                    err_px=err,
                    err_model_px=err_model,
                )
            )
        if args.overlay_dir is not None and overlays_saved < args.overlay_max:
            out_name = (
                f"{overlays_saved:04d}_{sample.sample_id}_{sample.phase}_"
                f"{sample.camera}_{sample.target_module_name}_{sample.port_name}.png"
            )
            draw_overlay(
                img,
                sample,
                gt_px,
                pred_px,
                confidence,
                args.overlay_dir.expanduser() / out_name,
            )
            overlays_saved += 1

    summary = summarize_rows(rows)
    if args.csv_prefix is not None:
        prefix = args.csv_prefix.expanduser()
        write_rows_csv(prefix.with_name(prefix.name + "_rows.csv"), rows)
        write_summary_csv(prefix.with_name(prefix.name + "_summary.csv"), summary)

    all_row = next(
        row
        for row in summary
        if row["group_type"] == "all" and row["group_id"] == "all"
    )
    print(f"Device: {device}")
    print(f"Model: {args.model.expanduser()}")
    print(f"Checkpoint epoch: {ckpt.get('epoch', '?')}")
    print(f"Labels: {args.labels.expanduser()} rows={len(samples)}")
    print(
        "All landmarks: "
        f"model_mean={all_row['err_model_mean_px']:.2f}px "
        f"model_p90={all_row['err_model_p90_px']:.2f}px "
        f"orig_mean={all_row['err_mean_px']:.2f}px "
        f"orig_p90={all_row['err_p90_px']:.2f}px "
        f"vis_acc={all_row['visibility_acc']:.1f}%"
    )
    print("Per landmark:")
    for row in summary:
        if row["group_type"] == "landmark":
            print(
                f"  {row['group_id']}: model_mean={row['err_model_mean_px']:.2f}px "
                f"model_p90={row['err_model_p90_px']:.2f}px "
                f"orig_mean={row['err_mean_px']:.2f}px "
                f"vis={row['visibility_acc']:.1f}%"
            )
    if args.overlay_dir is not None:
        print(f"Saved {overlays_saved} overlays to {args.overlay_dir.expanduser()}")
    if args.csv_prefix is not None:
        print(f"Wrote CSVs with prefix {args.csv_prefix.expanduser()}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SC port heatmap detector")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument(
        "--csv-prefix", "--csv_prefix", dest="csv_prefix", type=Path, default=None
    )
    parser.add_argument(
        "--overlay-dir", "--overlay_dir", dest="overlay_dir", type=Path, default=None
    )
    parser.add_argument(
        "--overlay-max", "--overlay_max", dest="overlay_max", type=int, default=50
    )
    parser.add_argument(
        "--max-samples", "--max_samples", dest="max_samples", type=int, default=0
    )
    parser.add_argument(
        "--visibility-threshold",
        "--visibility_threshold",
        dest="visibility_threshold",
        type=float,
        default=0.5,
    )
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()
