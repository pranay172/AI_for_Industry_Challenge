#!/usr/bin/env python3
"""Build SC port labels from GT-on policy captures.

It reads the `.json` + `.npz` files produced by `aic_model.policy` captures
and emits a JSONL manifest for `train_sc_port_detector.py`.

Unlike SFP, each SC module exposes one task port (`sc_port_base`). The capture
metadata already stores the GT projected visible face center and face corners,
so this script preserves those labels directly instead of inferring additional
asset geometry from another object.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from PIL import Image, ImageDraw
except ImportError:
    Image = None
    ImageDraw = None


CAMERAS = ("left", "center", "right")
ASSET_REL_PATH = "aic_assets/models/SC Port/model.sdf"
ASSET_GEOMETRY_VERSION = "sc_port_base_v1"
SC_PORT_NAME = "sc_port_base"
SC_FACE_WIDTH_M = 0.010
SC_FACE_HEIGHT_M = 0.025
SC_FACE_Z_PORT_M = -0.01564


def points_from_flat_norm(
    corners_norm: list[float],
    image_size: tuple[int, int],
) -> tuple[list[list[float]], list[list[float]], list[bool]]:
    width, height = image_size
    points_norm: list[list[float]] = []
    points_px: list[list[float]] = []
    visible: list[bool] = []
    for idx in range(0, min(len(corners_norm), 8), 2):
        try:
            u_norm = float(corners_norm[idx])
            v_norm = float(corners_norm[idx + 1])
        except (TypeError, ValueError, IndexError):
            u_norm = -1.0
            v_norm = -1.0
        u_px = u_norm * max(width, 1)
        v_px = v_norm * max(height, 1)
        is_visible = 0.0 <= u_norm <= 1.0 and 0.0 <= v_norm <= 1.0
        points_norm.append([u_norm, v_norm])
        points_px.append([u_px, v_px])
        visible.append(is_visible)
    return points_norm, points_px, visible


def project_camera_point(
    point_camera_m: list[float],
    intrinsics_k: list[float],
    image_size: tuple[int, int],
) -> dict:
    width, height = image_size
    if len(point_camera_m) < 3 or len(intrinsics_k) < 9:
        return {
            "points_camera_m": [point_camera_m],
            "points_px": [[-1.0, -1.0]],
            "points_norm": [[-1.0, -1.0]],
            "visible": [False],
            "all_visible": False,
            "any_visible": False,
        }
    x, y, z = [float(v) for v in point_camera_m[:3]]
    if z <= 1e-9:
        u = -1.0
        v = -1.0
        visible = False
    else:
        u = float(intrinsics_k[0]) * (x / z) + float(intrinsics_k[2])
        v = float(intrinsics_k[4]) * (y / z) + float(intrinsics_k[5])
        visible = 0.0 <= u < width and 0.0 <= v < height
    return {
        "points_camera_m": [[x, y, z]],
        "points_px": [[u, v]],
        "points_norm": [[u / max(width, 1), v / max(height, 1)]],
        "visible": [visible],
        "all_visible": bool(visible),
        "any_visible": bool(visible),
    }


def face_center_from_gt(gt: dict, image_size: tuple[int, int]) -> dict:
    width, height = image_size
    try:
        u = float(gt.get("u", -1.0))
        v = float(gt.get("v", -1.0))
    except (TypeError, ValueError):
        u = -1.0
        v = -1.0
    visible = bool(gt.get("visible", False)) and 0.0 <= u < width and 0.0 <= v < height
    return {
        "points_camera_m": [gt.get("xyz_camera", [])],
        "points_px": [[u, v]],
        "points_norm": [[u / max(width, 1), v / max(height, 1)]],
        "visible": [visible],
        "all_visible": visible,
        "any_visible": visible,
    }


def face_corners_from_gt(gt: dict, image_size: tuple[int, int]) -> Optional[dict]:
    corners_norm = gt.get("corners_norm", [])
    if len(corners_norm) < 8:
        return None
    points_norm, points_px, visible = points_from_flat_norm(corners_norm, image_size)
    return {
        "points_camera_m": [],
        "points_px": points_px,
        "points_norm": points_norm,
        "visible": visible,
        "all_visible": bool(all(visible)),
        "any_visible": bool(any(visible)),
    }


def build_candidate_label(
    meta: dict,
    camera: str,
    gt: dict,
    label_category: str,
    occlusion_state: str,
) -> Optional[dict]:
    if not gt.get("visible", False):
        return None
    task = meta.get("task", {})
    if str(task.get("plug_type", "")).lower() != "sc":
        return None
    if str(task.get("port_type", "")).lower() != "sc":
        return None
    selected_port = str(task.get("port_name", ""))
    if selected_port != SC_PORT_NAME:
        return None

    image_size_raw = gt.get("image_size", [])
    if len(image_size_raw) < 2:
        return None
    image_size = (int(image_size_raw[0]), int(image_size_raw[1]))

    face_corners = face_corners_from_gt(gt, image_size)
    if face_corners is None:
        return None
    face_center = face_center_from_gt(gt, image_size)
    port_origin = project_camera_point(
        gt.get("xyz_camera_port_origin", []),
        gt.get("intrinsics_k", []),
        image_size,
    )

    sample_id = str(meta.get("sample_id", ""))
    row = {
        "sample_id": sample_id,
        "episode_id": meta.get("episode_id", ""),
        "scene_id": meta.get("scene_id", ""),
        "npz_path": "",
        "image_key": f"{camera}_image",
        "camera": camera,
        "phase": str(meta.get("phase", "")),
        "task": {
            "id": str(task.get("id", "")),
            "plug_type": str(task.get("plug_type", "")),
            "port_type": str(task.get("port_type", "")),
            "port_name": selected_port,
            "target_module_name": str(task.get("target_module_name", "")),
        },
        "candidate": {
            "instance_name": str(task.get("target_module_name", "")),
            "available_port_names": [SC_PORT_NAME],
            "selected_port_name": selected_port,
            "source": "gt_target_port_tf",
        },
        "asset": {
            "geometry_version": ASSET_GEOMETRY_VERSION,
            "relative_path": ASSET_REL_PATH,
            "face_width_m": SC_FACE_WIDTH_M,
            "face_height_m": SC_FACE_HEIGHT_M,
            "face_z_port_m": SC_FACE_Z_PORT_M,
        },
        "image_size": list(image_size),
        "visibility": {
            "target_port_gt_visible": bool(gt.get("visible", False)),
            "occlusion": occlusion_state,
        },
        "label_category": label_category,
        "sc_port": {
            "name": SC_PORT_NAME,
            "is_task_target": True,
            "face_corner_order": ["TL", "TR", "BR", "BL"],
            "face_corners": face_corners,
            "face_center": face_center,
            "port_origin": port_origin,
            "bbox_width_px": float(gt.get("bbox_width_px", 0.0)),
            "bbox_height_px": float(gt.get("bbox_height_px", 0.0)),
            "intrinsics_k": gt.get("intrinsics_k", []),
            "R_cam_from_port": gt.get("R_cam_from_port", []),
            "xyz_camera_port_origin": gt.get("xyz_camera_port_origin", []),
        },
    }

    plug_tip = meta.get("plug_tip", {}).get(camera)
    if isinstance(plug_tip, dict):
        try:
            tip_u = float(plug_tip.get("u", -1.0))
            tip_v = float(plug_tip.get("v", -1.0))
        except (TypeError, ValueError):
            tip_u = -1.0
            tip_v = -1.0
        row["plug_tip"] = {
            "source": "tf_frame_origin",
            "frame": str(plug_tip.get("frame", "")),
            "visible": bool(plug_tip.get("visible", False)),
            "points_camera_m": [plug_tip.get("xyz_camera", [])],
            "points_px": [[tip_u, tip_v]],
            "points_norm": [
                [
                    tip_u / max(float(image_size[0]), 1.0),
                    tip_v / max(float(image_size[1]), 1.0),
                ]
            ],
        }
    return row


def draw_overlay(image: np.ndarray, label: dict, output_path: Path) -> bool:
    if Image is None or ImageDraw is None:
        return False
    img = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(img)

    port = label["sc_port"]
    color = (80, 220, 160)
    corners = port["face_corners"]["points_px"]
    if all(p[0] >= 0 and p[1] >= 0 for p in corners):
        draw.line(
            [tuple(p) for p in corners] + [tuple(corners[0])],
            fill=color,
            width=2,
        )
        for name, point in zip(port["face_corner_order"], corners):
            u, v = point
            draw.ellipse([u - 3, v - 3, u + 3, v + 3], fill=color)
            draw.text((u + 5, v - 4), name, fill=color)

    center = port["face_center"]["points_px"][0]
    u, v = center
    if u >= 0 and v >= 0:
        draw.ellipse([u - 5, v - 5, u + 5, v + 5], fill=(255, 80, 80))
        draw.text(
            (u + 7, v - 6),
            f"{label['task']['target_module_name']} {SC_PORT_NAME}",
            fill=(255, 80, 80),
        )

    origin = port["port_origin"]["points_px"][0]
    ou, ov = origin
    if ou >= 0 and ov >= 0:
        draw.ellipse([ou - 3, ov - 3, ou + 3, ov + 3], fill=(80, 160, 255))
        draw.text((ou + 5, ov - 4), "port_origin", fill=(80, 160, 255))

    plug_tip = label.get("plug_tip", {})
    if plug_tip.get("visible", False):
        points = plug_tip.get("points_px", [])
        if points:
            tu, tv = points[0]
            if tu >= 0 and tv >= 0:
                draw.line([tu - 6, tv, tu + 6, tv], fill=(255, 220, 40), width=2)
                draw.line([tu, tv - 6, tu, tv + 6], fill=(255, 220, 40), width=2)
                draw.text((tu + 7, tv - 6), "plug_tip", fill=(255, 220, 40))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    return True


def iter_capture_jsons(capture_dirs: list[Path]):
    seen: set[Path] = set()
    for capture_dir in capture_dirs:
        for json_path in sorted(capture_dir.expanduser().glob("*.json")):
            resolved = json_path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            yield json_path


def build_manifest(
    capture_dirs: list[Path],
    output_path: Path,
    occluded_output_path: Optional[Path],
    overlay_dir: Optional[Path],
    overlay_max: int,
    include_phases: set[str],
    occlusion_phases: set[str],
) -> tuple[int, int, int, int, dict[str, int]]:
    rows = []
    occluded_rows = []
    overlays_saved = 0
    overlay_warned = False
    skipped_by_phase = 0
    skip_counts = {
        "missing_npz": 0,
        "not_sc": 0,
        "gt_not_visible_or_incomplete": 0,
    }

    for json_path in iter_capture_jsons(capture_dirs):
        try:
            meta = json.loads(json_path.read_text())
        except Exception as exc:
            print(f"WARNING: skipping {json_path}: {exc}")
            continue
        phase = str(meta.get("phase", ""))
        is_clean = phase in include_phases
        is_occluded = phase in occlusion_phases
        if not is_clean and not (occluded_output_path is not None and is_occluded):
            skipped_by_phase += 1
            continue

        npz_path = json_path.with_suffix(".npz")
        if not npz_path.exists():
            skip_counts["missing_npz"] += 1
            continue

        task = meta.get("task", {})
        if str(task.get("plug_type", "")).lower() != "sc":
            skip_counts["not_sc"] += 1
            continue

        images = None
        for camera in CAMERAS:
            gt = meta.get("ground_truth", {}).get(camera, {})
            label_category = (
                "frustum_training" if is_clean else "occluded_servo_validation"
            )
            occlusion_state = "phase_likely_occluded" if is_occluded else "not_marked"
            label = build_candidate_label(
                meta,
                camera,
                gt,
                label_category,
                occlusion_state,
            )
            if label is None:
                skip_counts["gt_not_visible_or_incomplete"] += 1
                continue
            label["npz_path"] = str(npz_path.resolve())
            if is_clean:
                rows.append(label)
            else:
                occluded_rows.append(label)

            if overlay_dir is not None and overlays_saved < overlay_max:
                if images is None:
                    try:
                        images = np.load(npz_path)
                    except Exception as exc:
                        print(f"WARNING: could not load {npz_path.name}: {exc}")
                        continue
                image_key = label["image_key"]
                if image_key not in images:
                    continue
                out_name = (
                    f"{overlays_saved:04d}_{json_path.stem}_{camera}_"
                    f"{label_category}_"
                    f"{label['task']['target_module_name']}_"
                    f"{label['task']['port_name']}.png"
                )
                ok = draw_overlay(images[image_key], label, overlay_dir / out_name)
                if ok:
                    overlays_saved += 1
                elif not overlay_warned:
                    print("WARNING: --overlay-dir requested but Pillow is unavailable")
                    overlay_warned = True

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    if occluded_output_path is not None:
        occluded_output_path.parent.mkdir(parents=True, exist_ok=True)
        with occluded_output_path.open("w") as f:
            for row in occluded_rows:
                f.write(json.dumps(row) + "\n")
    return len(rows), len(occluded_rows), overlays_saved, skipped_by_phase, skip_counts


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build SC port JSONL labels from GT-on captures."
    )
    parser.add_argument(
        "capture_dirs",
        nargs="*",
        type=Path,
        help="Capture directories containing policy capture .json/.npz files.",
    )
    parser.add_argument(
        "--capture_dir",
        type=Path,
        action="append",
        default=[],
        help="Additional capture directory. May be passed more than once.",
    )
    parser.add_argument(
        "--output",
        "--out",
        dest="output",
        type=Path,
        required=True,
        help="Output JSONL manifest path for phase-selected rows; occlusion is unverified.",
    )
    parser.add_argument(
        "--occluded-output",
        "--occluded_output",
        dest="occluded_output",
        type=Path,
        default=None,
        help=(
            "Optional JSONL path for occlusion/servo-validation rows. "
            "When omitted, occlusion phases are skipped."
        ),
    )
    parser.add_argument(
        "--include-phases",
        "--include_phases",
        dest="include_phases",
        nargs="+",
        default=["find_target", "coarse_align"],
        help=(
            "Phases to include in the clean training manifest. Defaults to "
            "find_target coarse_align."
        ),
    )
    parser.add_argument(
        "--occlusion-phases",
        "--occlusion_phases",
        dest="occlusion_phases",
        nargs="+",
        default=["pre_insert", "insert", "settle"],
        help=(
            "Phases treated as likely occluded. These are written only when "
            "--occluded-output is provided."
        ),
    )
    parser.add_argument(
        "--overlay-dir",
        "--overlay_dir",
        dest="overlay_dir",
        type=Path,
        default=None,
        help="Optional directory for label overlay PNGs.",
    )
    parser.add_argument(
        "--overlay-max",
        "--overlay_max",
        dest="overlay_max",
        type=int,
        default=50,
        help="Maximum overlays to save when --overlay-dir is set.",
    )
    args = parser.parse_args()
    args.capture_dirs = [*args.capture_dirs, *args.capture_dir]
    if not args.capture_dirs:
        parser.error("at least one capture directory is required")
    return args


def main() -> None:
    args = parse_args()
    n_rows, n_occluded_rows, n_overlays, n_skipped, skip_counts = build_manifest(
        [path.expanduser() for path in args.capture_dirs],
        args.output.expanduser(),
        (
            args.occluded_output.expanduser()
            if args.occluded_output is not None
            else None
        ),
        args.overlay_dir.expanduser() if args.overlay_dir is not None else None,
        args.overlay_max,
        set(args.include_phases),
        set(args.occlusion_phases),
    )
    print(f"Wrote {n_rows} frustum-filtered SC port label rows (occlusion unverified) to {args.output.expanduser()}")
    if args.occluded_output is not None:
        print(
            f"Wrote {n_occluded_rows} occlusion/servo label rows to "
            f"{args.occluded_output.expanduser()}"
        )
    if n_skipped:
        print(f"Skipped {n_skipped} capture files by phase filter")
    if any(skip_counts.values()):
        print(f"Other skipped rows/files: {skip_counts}")
    if args.overlay_dir is not None:
        print(f"Saved {n_overlays} overlays to {args.overlay_dir.expanduser()}")


if __name__ == "__main__":
    main()
