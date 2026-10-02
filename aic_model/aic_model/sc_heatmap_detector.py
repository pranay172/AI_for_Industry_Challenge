"""Runtime helpers for the SC port heatmap detector."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from .vision_runtime import transforms

LANDMARK_NAMES = (
    "sc_face_tl",
    "sc_face_tr",
    "sc_face_br",
    "sc_face_bl",
    "sc_face_center",
)

SC_PORT_NAME = "sc_port_base"
SC_FACE_WIDTH_M = 0.010
SC_FACE_HEIGHT_M = 0.025
SC_FACE_Z_PORT_M = -0.01564


def sc_face_corners_port() -> np.ndarray:
    """Return SC face corners in sc_port_base_link coordinates.

    Corner order matches the offline GT labels: TL, TR, BR, BL in image-space
    for the canonical camera view. SC uses port X as vertical and port Y as
    horizontal, so width and height are intentionally swapped in port axes.
    """
    hw = 0.5 * SC_FACE_WIDTH_M
    hh = 0.5 * SC_FACE_HEIGHT_M
    return np.array(
        [
            [-hh, +hw, SC_FACE_Z_PORT_M],
            [-hh, -hw, SC_FACE_Z_PORT_M],
            [+hh, -hw, SC_FACE_Z_PORT_M],
            [+hh, +hw, SC_FACE_Z_PORT_M],
        ],
        dtype=np.float64,
    )


def sc_face_center_port() -> np.ndarray:
    return np.array([0.0, 0.0, SC_FACE_Z_PORT_M], dtype=np.float64)


def sc_landmarks_port() -> np.ndarray:
    return np.vstack([sc_face_corners_port(), sc_face_center_port()[None, :]])


from .landmark_network import LandmarkHeatmapNet as ScPortHeatmapNet, heatmap_argmax
from .sc_face_decoder import TEMPLATE_YAW_LIMITS_RAD



class ScPortHeatmapRuntime:
    def __init__(
        self,
        model: ScPortHeatmapNet,
        img_size: int,
        heatmap_size: int,
        device: torch.device,
    ):
        self.model = model
        self.img_size = img_size
        self.heatmap_size = heatmap_size
        self.device = device
        self.decoder = "argmax_v1"
        self.preprocessing = "full_frame_v1"
        self.transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def infer(self, image: np.ndarray, support_mask=None, face_templates=None, rail_hull=None) -> dict | None:
        self.last_rejection_reason = None
        self.last_decoder_diagnostics = None
        img = Image.fromarray(image).convert("RGB")
        tensor = self.transform(img)
        if self.preprocessing == 'rail_conditioned_v1':
            from .rail_conditioning import append_rail_channel
            tensor = append_rail_channel(tensor, rail_hull)
        elif rail_hull is not None:
            raise ValueError('Rail channel requires rail_conditioned_v1 preprocessing')
        tensor = tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred_heatmaps, pred_vis = self.model(tensor)
            heatmaps_prob = torch.sigmoid(pred_heatmaps)
            card_decoder = self.decoder == "sfp_card_template_v1"
            template_decoder = self.decoder in TEMPLATE_YAW_LIMITS_RAD or card_decoder
            if self.decoder == "rail_local_log_quadratic_v1" or template_decoder:
                if support_mask is None:
                    raise ValueError("Rail decoder requires projected rail support")
                support = torch.as_tensor(support_mask, device=self.device)
                if support.dtype != torch.bool or tuple(support.shape) != tuple(heatmaps_prob.shape[-2:]) or not support.any():
                    raise ValueError("Invalid or empty rail support mask")
                heatmaps_prob = heatmaps_prob.masked_fill(~support[None, None], 0.)
            elif support_mask is not None:
                raise ValueError("Support masks require the explicit rail decoder")
            if template_decoder:
                if card_decoder:
                    from .sfp_face_decoder import decode_card as decode, CORNER_INDICES as corners
                else:
                    from .sc_face_decoder import decode_face as decode
                    corners = (0, 1, 2, 3)
                diagnostics = {}
                self.last_decoder_diagnostics = diagnostics
                face = decode(heatmaps_prob[0].cpu().numpy(), support.cpu().numpy(), face_templates, diagnostics)
                if face is None:
                    self.last_rejection_reason = diagnostics.get('reason', 'inconsistent_face')
                    return None
                pred_pts_hm = face['points']
                heatmap_conf = face['heatmap_confidence']
                rail_translation = face['translation']
                rail_yaw = face['yaw']
                refined_yaw = face.get('refined_yaw', rail_yaw)
            else:
                pred_pts_hm = heatmap_argmax(heatmaps_prob, refine=self.decoder in {
                    "local_log_quadratic_v1", "rail_local_log_quadratic_v1"})[0].cpu().numpy()
                heatmap_conf = heatmaps_prob.reshape(1, len(LANDMARK_NAMES), -1).max(dim=-1)[0][0].cpu().numpy()
            vis_prob = torch.sigmoid(pred_vis)[0].cpu().numpy()
            if template_decoder:
                # Corner channels are pooled, so no original channel identity
                # supplies a reliable per-corner visibility probability.
                vis_prob = np.full(len(vis_prob), np.min(vis_prob[list(corners)]))

        sx = img.width / float(self.heatmap_size)
        sy = img.height / float(self.heatmap_size)
        pred_px = np.column_stack([pred_pts_hm[:, 0] * sx, pred_pts_hm[:, 1] * sy])
        pred_norm = pred_px / np.array([img.width, img.height], dtype=np.float64)
        confidence = np.sqrt(
            np.clip(vis_prob, 0.0, 1.0) * np.clip(heatmap_conf, 0.0, 1.0)
        )
        result = {
            "points_px": pred_px.astype(np.float64),
            "points_norm": pred_norm.astype(np.float64),
            "confidence": confidence.astype(np.float64),
            "visibility_probability": vis_prob.astype(np.float64),
            "heatmap_confidence": heatmap_conf.astype(np.float64),
            "image_size": (img.width, img.height),
        }
        if template_decoder:
            result["rail_translation"] = float(rail_translation)
            result["rail_yaw"] = float(rail_yaw)
            result["rail_yaw_refined"] = float(refined_yaw)
        return result


def load_sc_port_heatmap(path: str | Path) -> ScPortHeatmapRuntime:
    ckpt_path = Path(path).expanduser()
    if not ckpt_path.exists():
        raise FileNotFoundError(str(ckpt_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    names = tuple(ckpt.get("landmark_names", LANDMARK_NAMES))
    if names != LANDMARK_NAMES:
        raise ValueError(
            f"checkpoint landmarks {names} do not match expected {LANDMARK_NAMES}"
        )
    input_channels = 4 if ckpt.get('preprocessing') == 'rail_conditioned_v1' else 3
    model = ScPortHeatmapNet(num_landmarks=len(LANDMARK_NAMES), input_channels=input_channels).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    img_size = int(ckpt.get("img_size", 224))
    heatmap_size = int(ckpt.get("heatmap_size", img_size // 4))
    runtime = ScPortHeatmapRuntime(model, img_size, heatmap_size, device)
    runtime.preprocessing = ckpt.get('preprocessing', 'full_frame_v1')
    runtime.decoder = ckpt.get('decoder', 'argmax_v1')
    if runtime.decoder not in {'argmax_v1', 'local_log_quadratic_v1', 'rail_local_log_quadratic_v1', *TEMPLATE_YAW_LIMITS_RAD}:
        raise ValueError(f'Unsupported decoder: {runtime.decoder}')
    if runtime.preprocessing not in {'full_frame_v1', 'target_crop_v1', 'rail_crop_v1', 'rail_conditioned_v1'}:
        raise ValueError(f'Unsupported preprocessing: {runtime.preprocessing}')
    if (runtime.decoder == 'rail_local_log_quadratic_v1' or runtime.decoder in TEMPLATE_YAW_LIMITS_RAD) and runtime.preprocessing not in {'rail_crop_v1', 'rail_conditioned_v1'}:
        raise ValueError('Rail decoder requires rail crop preprocessing')
    return runtime
