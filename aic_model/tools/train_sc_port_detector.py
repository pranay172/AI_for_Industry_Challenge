#!/usr/bin/env python3
"""Train an SC port or SFP face heatmap detector.

Input is a JSONL manifest from `label_sc_port_dataset.py` (`--landmarks sc`:
SC face corners TL, TR, BR, BL and the face center) or from
`scripts/label_sfp_faces.py` (`--landmarks sfp_faces`: both SFP port faces).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the aic_model package

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from aic_model.vision_runtime import transforms

LANDMARK_NAMES = (
    "sc_face_tl",
    "sc_face_tr",
    "sc_face_br",
    "sc_face_bl",
    "sc_face_center",
)
CORNER_INDICES = (0, 1, 2, 3)
CENTER_INDEX = 4


@dataclass(frozen=True)
class LandmarkSpec:
    """Landmark semantics shared by labels, training metrics and checkpoints."""
    names: tuple
    corner_indices: tuple
    center_indices: tuple
    model_type: str


SC_SPEC = LandmarkSpec(LANDMARK_NAMES, CORNER_INDICES, (CENTER_INDEX,), "ScPortHeatmapNet")
# Both SFP port faces of the requested NIC card: four corners then the center,
# sfp_port_0 first. Rows carry them in a generic 'landmarks' block.
SFP_FACES_SPEC = LandmarkSpec(
    tuple(f"sfp_port_{p}_{n}" for p in (0, 1) for n in ("tl", "tr", "br", "bl", "center")),
    (0, 1, 2, 3, 5, 6, 7, 8), (4, 9), "SfpFaceHeatmapNet")
SPECS = {"sc": SC_SPEC, "sfp_faces": SFP_FACES_SPEC}


from aic_model.dataset import (resolve_capture, capture_group, grouped_split,
                               validate_captures, split_manifest, assert_independent, load_rgb, target_crop, training_record, save_checkpoint, parse_runtime_crop, rail_crop)


@dataclass(frozen=True)
class ScPortSample:
    npz_path: str
    image_key: str
    sample_id: str
    phase: str
    camera: str
    target_module_name: str
    port_name: str
    points_norm: tuple[tuple[float, float], ...]
    visible: tuple[bool, ...]
    group_id: str = ""
    crop_box: tuple[int, int, int, int] | None = None
    crop_capture_sha256: str = ""
    rail_hull: tuple = ()


def _point_norm_from_projection(projection: dict) -> tuple[float, float]:
    points = projection.get("points_norm", [])
    if not points:
        return (-1.0, -1.0)
    return (float(points[0][0]), float(points[0][1]))


def _visible_from_projection(projection: dict) -> bool:
    visible = projection.get("visible", [])
    if not visible:
        return False
    return bool(visible[0])


def load_manifest(path: Path, spec: LandmarkSpec | None = None) -> list[ScPortSample]:
    rows: list[ScPortSample] = []
    with path.expanduser().open() as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc

            generic = row.get("landmarks")
            if generic is not None:
                points = [tuple(map(float, point)) for point in generic["points_norm"]]
                visible = [bool(value) for value in generic["visible"]]
                if spec is not None and tuple(generic["names"]) != spec.names:
                    raise ValueError(f"Landmark names differ from {spec.model_type} at {path}:{line_no}")
            else:
                points, visible = _sc_points(row, path, line_no)
            if spec is not None and len(points) != len(spec.names):
                raise ValueError(f"Expected {len(spec.names)} landmarks at {path}:{line_no}")
            rows.append(_sample(row, path, points, visible))
    return rows


def _sc_points(row, path, line_no):
    sc_port = row.get("sc_port", {})
    corners = sc_port.get("face_corners", {})
    corner_points = corners.get("points_norm", [])
    corner_visible = corners.get("visible", [])
    center = sc_port.get("face_center", {})
    if len(corner_points) < 4 or len(corner_visible) < 4:
        raise ValueError(f"Missing SC landmarks at {path}:{line_no}")
    points = [tuple(map(float, corner_points[i])) for i in range(4)]
    visible = [bool(corner_visible[i]) for i in range(4)]
    points.append(_point_norm_from_projection(center))
    visible.append(_visible_from_projection(center))
    return points, visible


def _sample(row, path, points, visible):
    return ScPortSample(
        npz_path=resolve_capture(row.get("npz_path", ""), path),
        group_id=capture_group(row, path),
        crop_box=parse_runtime_crop(row),
        rail_hull=tuple(tuple(point) for point in row.get("runtime_crop", {}).get("rail_hull_norm", [])),
        crop_capture_sha256=row.get("runtime_crop", {}).get("capture_sha256", ""),
        image_key=str(row.get("image_key", "")),
        sample_id=str(row.get("sample_id", "")),
        phase=str(row.get("phase", "")),
        camera=str(row.get("camera", "")),
        target_module_name=str(row.get("task", {}).get("target_module_name", "")),
        port_name=str(row.get("task", {}).get("port_name", "")),
        points_norm=tuple(points),
        visible=tuple(visible),
    )


def split_by_sample_id(
    samples: list[ScPortSample],
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Compatibility name; split whole scenes/episodes, never individual frames."""
    return grouped_split(samples, val_fraction, seed)


class ScPortHeatmapDataset(Dataset):
    def __init__(
        self,
        samples: list[ScPortSample],
        img_size: int,
        heatmap_size: int,
        augment: bool,
        preprocessing: str = "full_frame_v1",
    ):
        self.samples = samples
        self.img_size = img_size
        self.heatmap_size = heatmap_size
        self.augment = augment
        if preprocessing not in {"full_frame_v1", "target_crop_v1", "rail_crop_v1", "rail_conditioned_v1"}:
            raise ValueError(f"Unsupported preprocessing: {preprocessing}")
        self.preprocessing = preprocessing
        if preprocessing in {"rail_crop_v1", "rail_conditioned_v1"} and any(s.crop_box is None for s in samples):
            raise ValueError("rail_crop_v1 requires prepared runtime crops for every sample")
        if preprocessing == 'rail_conditioned_v1':
            from aic_model.rail_conditioning import validate_hull
            for sample in samples:
                validate_hull(sample.rail_hull)
        self.resize = transforms.Resize((img_size, img_size))
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        self.color_jitter = transforms.ColorJitter(
            brightness=0.25,
            contrast=0.25,
            saturation=0.15,
            hue=0.03,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, sample: ScPortSample) -> Image.Image:
        return load_rgb(sample)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        img = self._load_image(sample)
        points = np.array(sample.points_norm, dtype=np.float32)
        visible = np.array(sample.visible, dtype=np.float32)

        if self.preprocessing in {"rail_crop_v1", "rail_conditioned_v1"}:
            img, points, visible = rail_crop(img, points, visible, sample.crop_box)
        if self.preprocessing == "target_crop_v1":
            img, points, visible = target_crop(img, points, visible, self.augment)
        if self.augment:
            img = self.color_jitter(img)

        img = self.resize(img)
        img_t = self.normalize(self.to_tensor(img))
        if self.preprocessing == 'rail_conditioned_v1':
            from aic_model.rail_conditioning import append_rail_channel
            img_t = append_rail_channel(img_t, sample.rail_hull)
        heatmaps = make_heatmaps(points, visible, self.heatmap_size)
        points_hm = points.copy()
        points_hm[:, 0] *= self.heatmap_size
        points_hm[:, 1] *= self.heatmap_size
        return (
            img_t,
            torch.from_numpy(heatmaps),
            torch.from_numpy(visible),
            torch.from_numpy(points_hm.astype(np.float32)),
        )


def make_heatmaps(
    points_norm: np.ndarray, visible: np.ndarray, heatmap_size: int
) -> np.ndarray:
    sigma = max(1.0, heatmap_size / 56.0 * 1.5)
    yy, xx = np.mgrid[0:heatmap_size, 0:heatmap_size].astype(np.float32)
    heatmaps = np.zeros(
        (len(points_norm), heatmap_size, heatmap_size), dtype=np.float32
    )
    for idx, ((u, v), is_visible) in enumerate(zip(points_norm, visible)):
        if not is_visible or u < 0.0 or v < 0.0:
            continue
        cx = float(u) * heatmap_size
        cy = float(v) * heatmap_size
        heatmaps[idx] = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma**2))
    return heatmaps


from aic_model.landmark_network import LandmarkHeatmapNet as ScPortHeatmapNet, heatmap_argmax



def compute_loss(
    pred_heatmaps: torch.Tensor,
    pred_vis_logits: torch.Tensor,
    target_heatmaps: torch.Tensor,
    target_visible: torch.Tensor,
    negative_heatmap_weight: float = 0.,
) -> torch.Tensor:
    heatmap_loss = nn.functional.mse_loss(
        torch.sigmoid(pred_heatmaps),
        target_heatmaps,
        reduction="none",
    )
    heatmap_loss = heatmap_loss * (1.0 + 20.0 * target_heatmaps)
    heatmap_loss = heatmap_loss.mean(dim=(2, 3))
    weights = target_visible + (1. - target_visible) * negative_heatmap_weight
    heatmap_loss = (heatmap_loss * weights).sum() / weights.sum().clamp_min(1.0)
    vis_loss = nn.functional.binary_cross_entropy_with_logits(
        pred_vis_logits,
        target_visible,
    )
    return heatmap_loss + 0.1 * vis_loss


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    heatmap_size: int,
    img_size: int,
    preprocessing: str = "full_frame_v1",
    negative_heatmap_weight: float = 0.,
    spec: LandmarkSpec = SC_SPEC,
) -> dict:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_errors = []
    err_sum = torch.zeros(len(spec.names), dtype=torch.float64)
    err_count = torch.zeros(len(spec.names), dtype=torch.float64)
    vis_correct = 0
    vis_total = 0
    with torch.no_grad():
        for imgs, heatmaps, visible, points_hm in loader:
            imgs = imgs.to(device)
            heatmaps = heatmaps.to(device)
            visible = visible.to(device)
            points_hm = points_hm.to(device)
            pred_heatmaps, pred_vis = model(imgs)
            loss = compute_loss(pred_heatmaps, pred_vis, heatmaps, visible, negative_heatmap_weight)
            total_loss += float(loss.item()) * imgs.shape[0]
            total_samples += imgs.shape[0]

            pred_pts = heatmap_argmax(torch.sigmoid(pred_heatmaps))
            err_hm = torch.linalg.norm(pred_pts - points_hm, dim=-1)
            err_px = err_hm * (img_size / heatmap_size)
            all_errors.extend(err_px[visible > 0.5].detach().cpu().tolist())
            for idx in range(len(spec.names)):
                mask = visible[:, idx] > 0.5
                if mask.any():
                    err_sum[idx] += err_px[mask, idx].double().sum().cpu()
                    err_count[idx] += int(mask.sum().item())

            pred_visible = torch.sigmoid(pred_vis) >= 0.5
            vis_correct += int((pred_visible == (visible > 0.5)).sum().item())
            vis_total += int(visible.numel())

    per_landmark = {}
    for idx, name in enumerate(spec.names):
        if err_count[idx] > 0:
            per_landmark[name] = float(err_sum[idx] / err_count[idx])
        else:
            per_landmark[name] = float("nan")
    valid_errs = [v for v in per_landmark.values() if math.isfinite(v)]
    corner_errs = [
        per_landmark[spec.names[idx]]
        for idx in spec.corner_indices
        if math.isfinite(per_landmark[spec.names[idx]])
    ]
    center_errs = [per_landmark[spec.names[idx]] for idx in spec.center_indices
                   if math.isfinite(per_landmark[spec.names[idx]])]
    center_px = sum(center_errs) / len(center_errs) if center_errs else float("nan")
    return {
        "coordinate_space": {"target_crop_v1":"resized_target_crop_pixels",
                             "rail_crop_v1":"resized_runtime_rail_crop_pixels",
                             "rail_conditioned_v1":"resized_runtime_rail_crop_pixels",
                             "full_frame_v1":"resized_full_frame_pixels"}[preprocessing],
        "p95_px": float(np.percentile(all_errors, 95)) if all_errors else float("nan"),
        "loss": total_loss / max(total_samples, 1),
        "mean_px": float(sum(valid_errs) / max(len(valid_errs), 1)),
        "corner_px": float(sum(corner_errs) / max(len(corner_errs), 1)),
        "center_px": float(center_px),
        "per_landmark_px": per_landmark,
        "visibility_acc": 100.0 * vis_correct / max(vis_total, 1),
    }


def load_backbone(model: nn.Module, checkpoint: dict, input_channels: int) -> None:
    """Copy every non-head tensor; new landmark semantics never reuse old heads."""
    from aic_model.rail_conditioning import initialize_state
    heads = ("heatmap_head.", "visibility_head.")
    state = {k: v for k, v in checkpoint['state_dict'].items() if not k.startswith(heads)}
    missing, unexpected = model.load_state_dict(initialize_state(state, input_channels), strict=False)
    if unexpected or not missing or any(not k.startswith(heads) for k in missing):
        raise ValueError(f"Backbone initialization mismatch: missing={missing} unexpected={unexpected}")


def train(args) -> None:
    if not math.isfinite(args.negative_heatmap_weight) or not 0 <= args.negative_heatmap_weight <= 1:
        raise ValueError('negative_heatmap_weight must be finite and between 0 and 1')
    manifest = Path(args.labels).expanduser()
    output = Path(args.output).expanduser()
    if any(output.with_suffix(suffix).exists() for suffix in ('.pt', '.split.json', '.run.json', '.history.jsonl')) or output.exists():
        raise ValueError(f"Checkpoint already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    run_record = training_record(args, __file__)
    output.with_suffix('.run.json').write_text(json.dumps(run_record, indent=2) + '\n')
    if args.img_size % 8 != 0:
        raise SystemExit("--img_size must be divisible by 8 for this encoder-decoder.")
    spec = SPECS[args.landmarks]
    samples = load_manifest(manifest, spec)
    if not samples:
        raise SystemExit(f"No SC port label rows found in {manifest}")
    if args.max_samples is not None and args.max_samples > 0:
        rng = random.Random(args.seed)
        if len(samples) > args.max_samples:
            samples = rng.sample(samples, args.max_samples)
            samples.sort(key=lambda sample: sample.sample_id)
        print(f"Using --max-samples={len(samples)}")

    validate_captures(samples)
    if args.validation_labels:
        from aic_model.dataset import explicit_validation_split
        validation_samples = load_manifest(args.validation_labels, spec)
        validate_captures(validation_samples)
        samples, train_idx, val_idx, split_record = explicit_validation_split(
            samples, validation_samples, manifest, args.validation_labels, args.seed)
    else:
        train_idx, val_idx = split_by_sample_id(samples, args.val_fraction, args.seed)
        split_record = split_manifest(samples, train_idx, val_idx, manifest, args.seed)
    output.with_suffix('.split.json').write_text(json.dumps(split_record, indent=2) + '\n')
    if not train_idx or not val_idx:
        raise SystemExit(
            "Train/validation split is empty; collect more labelled samples."
        )

    heatmap_size = args.img_size // 4
    train_ds_full = ScPortHeatmapDataset(
        samples, args.img_size, heatmap_size, augment=True, preprocessing=args.preprocessing
    )
    val_ds_full = ScPortHeatmapDataset(
        samples, args.img_size, heatmap_size, augment=False, preprocessing=args.preprocessing
    )
    train_ds = Subset(train_ds_full, train_idx)
    val_ds = Subset(val_ds_full, val_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(0, min(args.num_workers, 2)),
        pin_memory=True,
    )
    eval_loader = None
    if args.eval_labels:
        eval_manifest = Path(args.eval_labels).expanduser()
        eval_samples = load_manifest(eval_manifest, spec)
        validate_captures(eval_samples)
        assert_independent(samples, eval_samples)
        if eval_samples:
            eval_ds = ScPortHeatmapDataset(
                eval_samples,
                args.img_size,
                heatmap_size,
                augment=False,
                preprocessing=args.preprocessing,
            )
            eval_loader = DataLoader(
                eval_ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=max(0, min(args.num_workers, 2)),
                pin_memory=True,
            )
            print(f"Independent eval labels: {eval_manifest} ({len(eval_ds)} rows)")
        else:
            print(f"WARNING: --eval_labels {eval_manifest} had no usable rows")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_channels = 4 if args.preprocessing == 'rail_conditioned_v1' else 3
    model = ScPortHeatmapNet(num_landmarks=len(spec.names), input_channels=input_channels).to(device)
    if args.initialize and args.initialize_backbone:
        raise ValueError("Choose --initialize or --initialize-backbone, not both")
    if args.initialize_backbone:
        load_backbone(model, torch.load(args.initialize_backbone, map_location=device, weights_only=True),
                      input_channels)
    if args.initialize:
        initial = torch.load(args.initialize, map_location=device, weights_only=True)
        if tuple(initial['landmark_names']) != tuple(spec.names):
            raise ValueError('Initialization checkpoint has different landmark semantics')
        from aic_model.rail_conditioning import initialize_state
        model.load_state_dict(initialize_state(initial['state_dict'], input_channels), strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"Device: {device}")
    print(f"Labels: {manifest}")
    print(f"Samples: train={len(train_ds)} val={len(val_ds)}")
    print(f"Image size: {args.img_size}  Heatmap size: {heatmap_size}")
    print(f"Landmarks: {', '.join(spec.names)}")

    baseline = evaluate(model, val_loader, device, heatmap_size, args.img_size, args.preprocessing, args.negative_heatmap_weight, spec)
    output.with_suffix('.baseline.json').write_text(json.dumps(baseline, indent=2) + '\n')
    print(f"Initial held-out error ({args.preprocessing}): {baseline['mean_px']:.3f}px")
    best_mean_px = baseline['mean_px'] if args.initialize else float('inf')  # new heads start untrained
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        train_loss = 0.0
        train_count = 0
        for imgs, heatmaps, visible, _points_hm in train_loader:
            imgs = imgs.to(device)
            heatmaps = heatmaps.to(device)
            visible = visible.to(device)
            optimizer.zero_grad()
            pred_heatmaps, pred_vis = model(imgs)
            loss = compute_loss(pred_heatmaps, pred_vis, heatmaps, visible, args.negative_heatmap_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += float(loss.item()) * imgs.shape[0]
            train_count += imgs.shape[0]
        scheduler.step()

        val = evaluate(model, val_loader, device, heatmap_size, args.img_size, args.preprocessing, args.negative_heatmap_weight, spec)
        with output.with_suffix('.history.jsonl').open('a') as history:
            history.write(json.dumps({'epoch': epoch, 'validation': val}) + '\n')
        eval_msg = ""
        if eval_loader is not None:
            eval_stats = evaluate(
                model, eval_loader, device, heatmap_size, args.img_size, args.preprocessing, args.negative_heatmap_weight,
                spec,
            )
            eval_msg = (
                f" eval_mean_px={eval_stats['mean_px']:.2f} "
                f"eval_center_px={eval_stats['center_px']:.2f} "
                f"eval_vis={eval_stats['visibility_acc']:.1f}%"
            )
        train_loss /= max(train_count, 1)
        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"train={train_loss:.5f} val={val['loss']:.5f} "
            f"mean_px={val['mean_px']:.2f} corner_px={val['corner_px']:.2f} "
            f"center_px={val['center_px']:.2f} "
            f"vis_acc={val['visibility_acc']:.1f}%{eval_msg} {elapsed:.1f}s"
        )

        if val["mean_px"] < best_mean_px:
            best_mean_px = val["mean_px"]
            checkpoint = {
                "state_dict": {
                    k: v.detach().cpu() for k, v in model.state_dict().items()
                },
                "img_size": args.img_size,
                "heatmap_size": heatmap_size,
                "landmark_names": spec.names,
                "epoch": epoch,
                "val_mean_px": val["mean_px"],
                "val_corner_px": val["corner_px"],
                "val_center_px": val["center_px"],
                "val_per_landmark_px": val["per_landmark_px"],
                "model_type": spec.model_type,
                "labels": str(manifest),
                "dataset_split": split_record,
                "training_run": run_record,
                "preprocessing": args.preprocessing,
                "decoder": "local_log_quadratic_v1",
                "negative_heatmap_weight": args.negative_heatmap_weight,
                "initialize": str(args.initialize) if args.initialize else None,
                "initialize_backbone": str(args.initialize_backbone) if args.initialize_backbone else None,
            }
            save_checkpoint(checkpoint, output)
            print(f"  saved best checkpoint to {output}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train SC port heatmap detector")
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="JSONL labels from label_sc_port_dataset.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("~/aic_models/sc_port_heatmap.pt"),
        help="Output checkpoint path.",
    )
    parser.add_argument(
        "--eval-labels",
        "--eval_labels",
        dest="eval_labels",
        type=Path,
        default=None,
        help=(
            "Optional independent JSONL eval manifest, for example the "
            "occluded/pre-insert label file."
        ),
    )
    parser.add_argument("--preprocessing", choices=("full_frame_v1", "target_crop_v1", "rail_crop_v1", "rail_conditioned_v1"),
                        default="full_frame_v1", help="Crop experiments require explicit opt-in.")
    parser.add_argument("--initialize", type=Path, default=None)
    parser.add_argument("--initialize-backbone", type=Path, default=None,
                        help="Copy every non-head tensor from another checkpoint (new landmark semantics).")
    parser.add_argument("--landmarks", choices=tuple(SPECS), default="sc",
                        help="Landmark semantics: SC port face, or both SFP port faces of a NIC card.")
    parser.add_argument("--validation-labels", type=Path, default=None,
                        help="Use a preassigned scene-disjoint validation manifest instead of a random grouped split.")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--negative-heatmap-weight", type=float, default=0.,
                        help="Optional background supervision for invisible landmark channels; use reviewed visibility labels.")
    parser.add_argument(
        "--batch-size", "--batch_size", dest="batch_size", type=int, default=32
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--img-size", "--img_size", dest="img_size", type=int, default=384
    )
    parser.add_argument(
        "--val-fraction", "--val_fraction", dest="val_fraction", type=float, default=0.2
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--num-workers", "--num_workers", dest="num_workers", type=int, default=4
    )
    parser.add_argument(
        "--max-samples",
        "--max_samples",
        dest="max_samples",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests; omit for normal training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
