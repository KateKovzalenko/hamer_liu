import os
import tempfile
from IPython import embed
import torch
import cv2
import numpy as np
import base64
from pathlib import Path
from typing import Dict, Any
import urllib.request
from ..utils_debug import save_img

from .base_processor import BaseProcessor

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.datasets.vitdet_dataset import ViTDetDataset
from hamer.utils import recursive_to
from hamer.utils.renderer import Renderer, cam_crop_to_full
from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy

from vitpose_model import ViTPoseModel

FINAL_MODEL = CACHE_DIR_HAMER + "/model_final_f05665.pkl"

ASSETS = [
    DEFAULT_CHECKPOINT,
    FINAL_MODEL
]

def _assets_ready(cache_dir: Path) -> bool:
    sentinel = cache_dir / ".assets_ready"
    if sentinel.exists():
        return True
    if not all(Path(asset).exists() for asset in ASSETS):
        return False
    # create sentinel if all expected files present
    sentinel.touch()
    return True

def _remove_tar_gz_files(cache_dir: str | Path) -> int:
    """
    Remove top-level .tar.gz files in cache_dir (non-recursive).
    Returns number of files removed.
    """
    cache_dir = Path(cache_dir)
    if not cache_dir.is_dir():
        return 0
    removed = 0
    for p in cache_dir.iterdir():  # non-recursive
        if p.is_file() and p.name.endswith(".tar.gz"):
            try:
                p.unlink()
                removed += 1
            except Exception:
                # ignore failures or log if desired
                pass
    return removed

def ensure_hamer_assets() -> None:
    cache_dir = Path(CACHE_DIR_HAMER)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if _assets_ready(cache_dir):
        return

    MODEL_URL = (
        "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/"
        "cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
    )
    print("Downloading final model for ViTDet human detector...")
    urllib.request.urlretrieve(MODEL_URL, FINAL_MODEL)

    print("Downloading HaMeR model and assets...")
    download_models(str(cache_dir))
    # create sentinel if download produced expected files
    _assets_ready(cache_dir)
    _remove_tar_gz_files(cache_dir)

class HamerProcessor(BaseProcessor):
    """Processor implementation using HaMeR 3D Hand Mesh Reconstruction."""

    def __init__(self, device: str = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # Download & load model
        ensure_hamer_assets()
            
        self.model, self.model_cfg = load_hamer(DEFAULT_CHECKPOINT)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Load human detector (ViTDet)
        from detectron2.config import LazyConfig
        import hamer
        cfg_path = Path(hamer.__file__).parent / "configs" / "cascade_mask_rcnn_vitdet_h_75ep.py"
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = FINAL_MODEL
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        self.detector = DefaultPredictor_Lazy(detectron2_cfg)

        # Load keypoint detector
        self.vitpose = ViTPoseModel(self.device)

        # Renderer setup
        self.renderer = Renderer(self.model_cfg, faces=self.model.mano.faces)
        self.LIGHT_BLUE = (0.65, 0.74, 0.86)
    
    def _project_vertices_to_pixels(
        self,
        vertices: torch.Tensor,          # (N, 3)
        cam_t,                           # (3,) can be numpy or torch
        focal_length: float,
        img_res: tuple[int, int]         # (W, H)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert HaMeR 3D verts (mm space) into Camera Space XYZ and Pixel Space XY.
        
        Returns:
            vertices_cam (np.ndarray): (N, 3) [X, Y, Z] in camera metric space.
            vertices_px (np.ndarray): (N, 2) [u_px, v_px] in image pixel space.
        """

        # ---------------------------
        # 0) Make focal_length safe
        # ---------------------------
        if isinstance(focal_length, torch.Tensor):
            focal_length = float(focal_length.detach().cpu())

        # ---------------------------
        # 1) Move vertices to CPU
        # --------------------------- 
        verts_cpu = vertices.detach().cpu()

        # ---------------------------
        # 2) Handle cam_t safely (numpy OR torch)
        # ---------------------------
        if isinstance(cam_t, np.ndarray):
            cam_t_np = cam_t
        else:
            cam_t_np = cam_t.detach().cpu().numpy()

        tx, ty, tz = cam_t_np

        # ---------------------------
        # 3) Apply camera translation
        # --------------------------- 
        # This results in the 3D coordinates in the camera frame
        X = verts_cpu[:, 0] + tx
        Y = -(verts_cpu[:, 1] + ty)
        Z = verts_cpu[:, 2] + tz + 1e-8   # avoid divide-by-zero

        W, H = img_res

        # ---------------------------
        # 4) Perspective projection
        # ---------------------------  
        u = focal_length * (X / Z) + (W / 2)
        v = focal_length * (Y / Z) + (H / 2)

        # ---------------------------
        # 5) Pack results
        # ---------------------------
        # 3D Camera Coordinates
        verts_cam = np.stack([X, Y, Z], axis=1)
        
        # 2D Pixel Coordinates
        verts_px = np.stack([u, v], axis=1)

        return verts_cam, verts_px

    # -------------------------
    # Internal processing logic
    # -------------------------
    def _process_image_np(self, image_np: np.ndarray) -> Dict[str, Any]:
        """Internal helper that processes a numpy image (BGR)."""
        img_cv2 = image_np.copy()
        img_rgb = img_cv2[:, :, ::-1]

        # --- Step 1: Detect humans ---
        det_out = self.detector(img_cv2)
        det_instances = det_out["instances"]
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
        if valid_idx.sum() == 0:
            return {"error": "No person detected"}

        pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores = det_instances.scores[valid_idx].cpu().numpy()

        # --- Step 2: Detect body keypoints (to locate hands) ---
        vitposes_out = self.vitpose.predict_pose(
            img_rgb, [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)]
        )

        bboxes, is_right = [], []
        for vitposes in vitposes_out:
            left_hand_keyp = vitposes["keypoints"][-42:-21]
            right_hand_keyp = vitposes["keypoints"][-21:]

            # Left hand
            valid = left_hand_keyp[:, 2] > 0.5
            if valid.sum() > 3:
                bbox = [
                    left_hand_keyp[valid, 0].min(),
                    left_hand_keyp[valid, 1].min(),
                    left_hand_keyp[valid, 0].max(),
                    left_hand_keyp[valid, 1].max(),
                ]
                bboxes.append(bbox)
                is_right.append(0)

            # Right hand
            valid = right_hand_keyp[:, 2] > 0.5
            if valid.sum() > 3:
                bbox = [
                    right_hand_keyp[valid, 0].min(),
                    right_hand_keyp[valid, 1].min(),
                    right_hand_keyp[valid, 0].max(),
                    right_hand_keyp[valid, 1].max(),
                ]
                bboxes.append(bbox)
                is_right.append(1)

        if not bboxes:
            return {"error": "No hands detected"}

        boxes = np.stack(bboxes)
        right = np.stack(is_right)

        # --- Step 3: Run HaMeR model ---
        dataset = ViTDetDataset(self.model_cfg, img_cv2, boxes, right)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

        all_results = {"hands": []}
        all_verts_list = []
        all_cam_t_list = []
        all_right_list = []

        for batch in dataloader:
            batch = recursive_to(batch, self.device)
            with torch.no_grad():
                out = self.model(batch)

            pred_cam = out["pred_cam"]
            multiplier = (2 * batch["right"] - 1)
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]

            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            scaled_focal_length = (
                self.model_cfg.EXTRA.FOCAL_LENGTH / self.model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            )

            pred_cam_t_full = cam_crop_to_full(
                pred_cam, box_center, box_size, img_size, scaled_focal_length
            ).detach().cpu().numpy()

            batch_size = pred_cam.shape[0]
            for n in range(batch_size):
                verts_n = out["pred_vertices"][n]  # CUDA tensor
                # Right hand: keep X
                # Left hand: flip X
                is_r_n = int(batch["right"][n].detach().item())
                verts_n[:, 0] = (2 * is_r_n - 1) * verts_n[:, 0]
                cam_t_n = pred_cam_t_full[n]
                img_res = (img_cv2.shape[1], img_cv2.shape[0])

                # --- Convert vertices to pixel coordinates ---
                # Unpack both the 3D Camera Space coords and 2D Pixel Space coords
                verts_cam_xyz, verts_pixel_xy = self._project_vertices_to_pixels(
                    vertices=verts_n,  # CUDA tensor
                    cam_t=cam_t_n,     # NumPy array OK
                    focal_length=scaled_focal_length,
                    img_res=img_res
                )
                
                all_results["hands"].append({
                    "is_right": bool(is_r_n),
                    "vertices_3d": verts_cam_xyz.tolist(),   # XYZ in Camera Space
                    "vertices_pixel": verts_pixel_xy.tolist(), # XY in Pixel Space
                    "camera_translation": cam_t_n.tolist(),
                })

                all_verts_list.append(verts_n.detach().cpu().numpy())  # convert for renderer
                all_cam_t_list.append(cam_t_n.copy())
                all_right_list.append(is_r_n)

        # --- Step 4: Render full-frame image ---
        if all_verts_list:
            render_fn = getattr(self.renderer, "render_rgba_multiple", None)
            h_full, w_full = img_cv2.shape[:2]
            render_res_hw = (int(w_full), int(h_full))
            scaled_focal_length_full = (
                self.model_cfg.EXTRA.FOCAL_LENGTH / self.model_cfg.MODEL.IMAGE_SIZE * max(h_full, w_full)
            )

            if render_fn:
                rendered_combined = render_fn(
                    all_verts_list,
                    cam_t=all_cam_t_list,
                    render_res=render_res_hw,
                    focal_length=scaled_focal_length_full,
                    is_right=all_right_list,
                    mesh_base_color=self.LIGHT_BLUE,
                    scene_bg_color=(1, 1, 1),
                )
            else:
                # fallback: manual rendering
                renders = []
                for verts_n, cam_t_n in zip(all_verts_list, all_cam_t_list):
                    try:
                        r = self.renderer(
                            verts_n, cam_t_n, img_cv2,
                            mesh_base_color=self.LIGHT_BLUE,
                            scene_bg_color=(1, 1, 1)
                        )
                        renders.append(r)
                    except Exception:
                        renders.append(np.zeros((h_full, w_full, 4), dtype=np.float32))
                rendered_combined = np.zeros((h_full, w_full, 4), dtype=np.float32)
                for r in renders:
                    alpha = r[..., 3:4].astype(np.float32)
                    if alpha.max() > 1.5:
                        alpha /= 255.0
                    rendered_combined[..., :3] = rendered_combined[..., :3] * (1 - alpha) + r[..., :3] * alpha
                    rendered_combined[..., 3:4] = np.clip(rendered_combined[..., 3:4] + alpha, 0, 1)

            # Resize / composite over original image
            h_r, w_r = rendered_combined.shape[:2]
            if (h_r, w_r) != (h_full, w_full):
                rendered_combined = cv2.resize(rendered_combined, (w_full, h_full), interpolation=cv2.INTER_LINEAR)

            full_img_rgb = img_cv2.astype(np.float32)[:, :, ::-1] / 255.0
            alpha = rendered_combined[..., 3:4].astype(np.float32)
            if alpha.max() > 1.5:
                alpha /= 255.0
            rgb = rendered_combined[..., :3].astype(np.float32)
            overlay = full_img_rgb * (1 - alpha) + rgb * alpha
            overlay = np.clip(overlay, 0.0, 1.0)
            final_render_bgr = (overlay[:, :, ::-1] * 255).astype(np.uint8)

            _, png_bytes = cv2.imencode(".png", final_render_bgr)
            rendered_b64 = base64.b64encode(png_bytes).decode("utf-8")
            all_results["full_frame_rendered_image_base64"] = rendered_b64

        return all_results
    
    def _process_video(self, file_path, sample_rate=100):
        """
        Process video file by sampling frames.
        - First frame: full hand processing with rendered image.
        - Subsequent frames: vertices + camera translation only.
        """
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {file_path}")

        frame_count = 0
        sampled_results = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break  # exit if no more frames
            frame_count += 1

            # Always process the first frame
            if frame_count == 1:
                full_result = self._process_image_np(frame)
                sampled_results.append(full_result)
                continue

            # Sample other frames at given interval
            if frame_count % sample_rate == 0:
                try:
                    # Run same processing as _process_image_np, but skip rendering
                    img_cv2 = frame.copy()
                    img_rgb = img_cv2[:, :, ::-1]

                    # Detect humans
                    det_out = self.detector(img_cv2)
                    det_instances = det_out["instances"]
                    valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
                    if valid_idx.sum() == 0:
                        sampled_results.append({"error": "No person detected"})
                        continue

                    pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
                    pred_scores = det_instances.scores[valid_idx].cpu().numpy()

                    # Keypoints
                    vitposes_out = self.vitpose.predict_pose(
                        img_rgb, [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)]
                    )

                    bboxes, is_right = [], []
                    for vitposes in vitposes_out:
                        left_hand_keyp = vitposes["keypoints"][-42:-21]
                        right_hand_keyp = vitposes["keypoints"][-21:]

                        # Left hand
                        valid = left_hand_keyp[:, 2] > 0.5
                        if valid.sum() > 3:
                            bbox = [
                                left_hand_keyp[valid, 0].min(),
                                left_hand_keyp[valid, 1].min(),
                                left_hand_keyp[valid, 0].max(),
                                left_hand_keyp[valid, 1].max(),
                            ]
                            bboxes.append(bbox)
                            is_right.append(0)

                        # Right hand
                        valid = right_hand_keyp[:, 2] > 0.5
                        if valid.sum() > 3:
                            bbox = [
                                right_hand_keyp[valid, 0].min(),
                                right_hand_keyp[valid, 1].min(),
                                right_hand_keyp[valid, 0].max(),
                                right_hand_keyp[valid, 1].max(),
                            ]
                            bboxes.append(bbox)
                            is_right.append(1)

                    if not bboxes:
                        sampled_results.append({"error": "No hands detected"})
                        continue

                    boxes = np.stack(bboxes)
                    right = np.stack(is_right)

                    dataset = ViTDetDataset(self.model_cfg, img_cv2, boxes, right)
                    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

                    frame_results = {"hands": []}
                    for batch in dataloader:
                        batch = recursive_to(batch, self.device)
                        with torch.no_grad():
                            out = self.model(batch)

                        pred_cam = out["pred_cam"]
                        multiplier = (2 * batch["right"] - 1)
                        pred_cam[:, 1] = multiplier * pred_cam[:, 1]

                        box_center = batch["box_center"].float()
                        box_size = batch["box_size"].float()
                        img_size = batch["img_size"].float()
                        scaled_focal_length = (
                            self.model_cfg.EXTRA.FOCAL_LENGTH / self.model_cfg.MODEL.IMAGE_SIZE * img_size.max()
                        )

                        pred_cam_t_full = cam_crop_to_full(
                            pred_cam, box_center, box_size, img_size, scaled_focal_length
                        ).detach().cpu().numpy()

                        verts = out["pred_vertices"][0].detach().cpu().numpy()
                        cam_t = pred_cam_t_full[0]
                        is_r = bool(batch["right"][0].cpu().numpy())

                        # For video frames after first, skip rendering
                        frame_results["hands"].append(
                            {
                                "is_right": is_r,
                                "vertices": verts.tolist(),
                                "camera_translation": cam_t.tolist(),
                            }
                        )

                    sampled_results.append(frame_results)

                except Exception as e:
                    sampled_results.append({"error": str(e)})

        cap.release()
        return {
            "frames_processed": len(sampled_results),
            "samples": sampled_results
        }


    # -------------------------
    # Flask interface methods
    # -------------------------
    def process_image_file(self, file):
        """Process an uploaded image file-like object (from Flask)."""
        filestr = file.read()
        npimg = np.frombuffer(filestr, np.uint8)
        image_np = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        return self._process_image_np(image_np)

    def process_video_file(self, file):
        """
        Process an uploaded video file-like object (from Flask).
        Saves the uploaded file temporarily and calls _process_video.
        """

        # Save uploaded file to a temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        try:
            result = self._process_video(tmp_path)
        finally:
            os.remove(tmp_path)

        return result