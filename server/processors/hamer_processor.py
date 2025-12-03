import os
import tempfile
import urllib.request
import base64
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional, Union

import torch
import cv2
import numpy as np

# Third-party library imports
from hamer.configs import CACHE_DIR_HAMER
from hamer.models import download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.datasets.vitdet_dataset import ViTDetDataset
from hamer.utils import recursive_to
from hamer.utils.renderer import Renderer, cam_crop_to_full
from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
from detectron2.config import LazyConfig
import hamer

# Local/Custom imports
# Assumed to exist based on original snippet
from .base_processor import BaseProcessor 
from vitpose_model import ViTPoseModel

# --- Configuration Management ---

@dataclass
class HamerConfig:
    """Immutable configuration for HamerProcessor."""
    device: str
    confidence_threshold: float = 0.5
    box_threshold: float = 0.25
    hand_keypoint_threshold: float = 0.5
    min_valid_keypoints: int = 3
    batch_size: int = 8
    light_blue: Tuple[float, float, float] = (0.65, 0.74, 0.86)
    
    # Model Paths
    final_model_filename: str = "model_final_f05665.pkl"
    model_url: str = (
        "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/"
        "cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
    )

    @property
    def cache_dir(self) -> Path:
        return Path(CACHE_DIR_HAMER)

    @property
    def final_model_path(self) -> Path:
        return self.cache_dir / self.final_model_filename

# --- Global Helper Functions ---

def _assets_ready(config: HamerConfig) -> bool:
    sentinel = config.cache_dir / ".assets_ready"
    assets = [DEFAULT_CHECKPOINT, str(config.final_model_path)]
    
    if sentinel.exists():
        return True
    if not all(Path(asset).exists() for asset in assets):
        return False
    sentinel.touch()
    return True

def _remove_tar_gz_files(cache_dir: Path) -> int:
    """Remove top-level .tar.gz files in cache_dir (non-recursive)."""
    if not cache_dir.is_dir():
        return 0
    removed = 0
    for p in cache_dir.iterdir():
        if p.is_file() and p.name.endswith(".tar.gz"):
            try:
                p.unlink()
                removed += 1
            except Exception:
                pass
    return removed

def ensure_hamer_assets(config: HamerConfig) -> None:
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    
    if _assets_ready(config):
        return

    print("Downloading final model for ViTDet human detector...")
    urllib.request.urlretrieve(config.model_url, str(config.final_model_path))

    print("Downloading HaMeR model and assets...")
    download_models(str(config.cache_dir))
    
    _assets_ready(config)
    _remove_tar_gz_files(config.cache_dir)

# --- Main Class ---

class HamerProcessor(BaseProcessor):
    """
    High-Performance Processor for HaMeR 3D Hand Mesh Reconstruction.
    Optimized for GPU execution and architectural separation of concerns.
    """

    def __init__(self, device: str = None):
        # Resolve device: Prioritize argument -> CUDA -> CPU
        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.config = HamerConfig(device=resolved_device)
        self.device = torch.device(self.config.device)
        
        ensure_hamer_assets(self.config)
        self._init_models()

    def _init_models(self):
        """Initialize all neural network models."""
        # 1. HaMeR Model
        self.model, self.model_cfg = load_hamer(DEFAULT_CHECKPOINT)
        self.model = self.model.to(self.device)
        self.model.eval()

        # 2. ViTDet (Human Detector)
        cfg_path = Path(hamer.__file__).parent / "configs" / "cascade_mask_rcnn_vitdet_h_75ep.py"
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = str(self.config.final_model_path)
        
        # Apply threshold to box predictors
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = self.config.box_threshold
            
        self.detector = DefaultPredictor_Lazy(detectron2_cfg)

        # 3. ViTPose (Keypoint Detector)
        self.vitpose = ViTPoseModel(self.device)

        # 4. Renderer
        self.renderer = Renderer(self.model_cfg, faces=self.model.mano.faces)

    # --------------------------------------------------------------------------
    # Core Pipeline Steps
    # --------------------------------------------------------------------------

    def _detect_humans(self, img_cv2: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Run Detectron2 to find humans."""
        # Detectron2 LazyConfig models expect BGR standard input
        det_out = self.detector(img_cv2)
        det_instances = det_out["instances"]
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > self.config.confidence_threshold)
        
        if valid_idx.sum() == 0:
            return None, None

        pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores = det_instances.scores[valid_idx].cpu().numpy()
        return pred_bboxes, pred_scores

    def _extract_hand_bboxes(self, img_rgb: np.ndarray, pred_bboxes: np.ndarray, pred_scores: np.ndarray) -> Tuple[List[Any], List[int]]:
        """Run ViTPose and extract hand bounding boxes from body keypoints."""
        # Batch preparation for ViTPose
        vitposes_out = self.vitpose.predict_pose(
            img_rgb, 
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)]
        )

        bboxes = []
        is_right = []

        for vitposes in vitposes_out:
            # Keypoint indices: Left hand [-42:-21], Right hand [-21:]
            left_hand_keyp = vitposes["keypoints"][-42:-21]
            right_hand_keyp = vitposes["keypoints"][-21:]

            for keypoints, is_r_flag in [(left_hand_keyp, 0), (right_hand_keyp, 1)]:
                valid = keypoints[:, 2] > self.config.hand_keypoint_threshold
                if valid.sum() > self.config.min_valid_keypoints:
                    # Calculate bounding box from valid keypoints
                    bbox = [
                        keypoints[valid, 0].min(),
                        keypoints[valid, 1].min(),
                        keypoints[valid, 0].max(),
                        keypoints[valid, 1].max(),
                    ]
                    bboxes.append(bbox)
                    is_right.append(is_r_flag)

        return bboxes, is_right

    def _run_hamer_inference(
        self, 
        img_cv2: np.ndarray, 
        boxes: np.ndarray, 
        right_flags: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Run the actual HaMeR model on extracted hand boxes."""
        dataset = ViTDetDataset(self.model_cfg, img_cv2, boxes, right_flags)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=0
        )

        inference_results = []

        for batch in dataloader:
            batch = recursive_to(batch, self.device)
            with torch.no_grad():
                out = self.model(batch)

            # Camera logic
            pred_cam = out["pred_cam"]
            multiplier = (2 * batch["right"] - 1)
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]
            
            # Unpack batch parameters for scaling
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            
            # Focal length calculation (scalar per image usually, but calculated batch-wise here)
            scaled_focal_length = (
                self.model_cfg.EXTRA.FOCAL_LENGTH / 
                self.model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            )

            # Convert crop camera parameters to full image space
            pred_cam_t_full = cam_crop_to_full(
                pred_cam, box_center, box_size, img_size, scaled_focal_length
            )

            batch_len = pred_cam.shape[0]
            for n in range(batch_len):
                verts_n = out["pred_vertices"][n] # Tensor on GPU
                is_r_n = int(batch["right"][n].detach().item())
                
                # Flip left hand vertices to match right hand canonical space if needed
                # Operation performed on GPU
                if is_r_n == 0:
                    verts_n[:, 0] = -1 * verts_n[:, 0]
                
                cam_t_n = pred_cam_t_full[n] # Tensor on GPU
                
                inference_results.append({
                    "vertices_cuda": verts_n,
                    "cam_t_cuda": cam_t_n,
                    "is_right": is_r_n,
                    "focal_length": scaled_focal_length # Scalar tensor
                })
        
        return inference_results

    # --------------------------------------------------------------------------
    # Projection Mathematics (Vectorized & GPU Optimized)
    # --------------------------------------------------------------------------

    def _project_vertices_to_pixels(
            self,
            vertices: torch.Tensor,
            cam_t: torch.Tensor,
            focal_length: float,
            img_res: tuple[int, int]
        ) -> torch.Tensor:
        """
        Convert 3D verts to Pixel Space XY.
        Operations performed entirely on GPU to avoid sync overhead.
        """
        # Ensure scalar is float, not Tensor
        if isinstance(focal_length, torch.Tensor):
            focal_length = focal_length.item()

        tx, ty, tz = cam_t[0], cam_t[1], cam_t[2]

        # 1. Apply Camera Translation (World -> Camera Frame Base)
        X_trans = vertices[:, 0] + tx
        Y_trans = vertices[:, 1] + ty
        Z_trans = vertices[:, 2] + tz

        # 2. Apply Rotation (RotX 180 degrees)
        # RotX(180) maps: (x, y, z) -> (x, -y, -z)
        X_view = X_trans
        Y_view = Y_trans
        Z_view = -Z_trans

        # 3. Perspective Projection
        # Depth is distance along the Z axis: depth = -Z_view
        depth = -Z_view + 1e-8 # Avoid division by zero
        
        W, H = img_res
        cx, cy = W / 2.0, H / 2.0

        # Standard Pinhole Projection: u = fx * (x / depth) + cx
        u = focal_length * (X_view / depth) + cx
        v = focal_length * (Y_view / depth) + cy

        return torch.stack([u, v], dim=1)

    def _project_vertices_to_planar_z0(
        self,
        vertices: torch.Tensor,
        cam_t: torch.Tensor,
        focal_length: float
    ) -> Tuple[torch.Tensor, float]:
        """
        Project 3D vertices onto the Z=0 plane while preserving perspective-based scaling.
        
        FIXED: scaling_factor is now returned as a scalar (mean scale) rather than a 
        per-vertex tensor, correcting the downstream serialization issue.
        """
        if isinstance(focal_length, torch.Tensor):
            focal_length = focal_length.item()

        tx, ty, tz = cam_t[0], cam_t[1], cam_t[2]

        # 1. Transform to Camera Space
        X = vertices[:, 0] + tx
        Y = (vertices[:, 1] + ty)
        Z = vertices[:, 2] + tz + 1e-8

        # 2. Calculate Perspective Scale Factor for X and Y
        # This determines how much to shrink/expand based on distance from camera.
        scale_factor_vec = focal_length / Z

        # 3. Apply Projection to X and Y
        X_planar = X * scale_factor_vec
        Y_planar = Y * scale_factor_vec
        
        # 4. Handle Z (Volume Preservation)
        # Calculate the mean depth of the hand
        Z_mean = torch.mean(Z)
        
        # Calculate the scalar scaling factor for the centroid
        scale_mean = focal_length / Z_mean
        
        # Center volume at 0 and scale by the mean scale
        Z_planar = (Z - Z_mean) * scale_mean

        # Stack X, Y, Z
        planar_verts = torch.stack([X_planar, Y_planar, Z_planar], dim=1)
        
        # Return the tensor and the scalar float for the whole object
        return planar_verts, scale_mean.item()

    # --------------------------------------------------------------------------
    # Rendering & Processing
    # --------------------------------------------------------------------------

    def _render_scene(
        self, 
        img_cv2: np.ndarray, 
        inference_results: List[Dict[str, Any]]
    ) -> str:
        """Render mesh overlay on the image and return Base64 string."""
        if not inference_results:
            return ""

        # Move to CPU only when strictly necessary for the renderer (if it's not GPU compatible)
        # Assuming Hamer renderer needs numpy arrays:
        all_verts = [res["vertices_cuda"].detach().cpu().numpy() for res in inference_results]
        all_cam_t = [res["cam_t_cuda"].detach().cpu().numpy() for res in inference_results]
        all_right = [res["is_right"] for res in inference_results]

        h_full, w_full = img_cv2.shape[:2]
        
        # Efficient batch rendering
        render_fn = getattr(self.renderer, "render_rgba_multiple", None)
        
        focal_length_val = inference_results[0]["focal_length"]
        if isinstance(focal_length_val, torch.Tensor):
            focal_length_val = focal_length_val.item()

        scaled_focal_length_full = (
            self.model_cfg.EXTRA.FOCAL_LENGTH / 
            self.model_cfg.MODEL.IMAGE_SIZE * max(h_full, w_full)
        )

        rendered_combined = None

        if render_fn:
            rendered_combined = render_fn(
                all_verts,
                cam_t=all_cam_t,
                render_res=(w_full, h_full),
                focal_length=scaled_focal_length_full,
                is_right=all_right,
                mesh_base_color=self.config.light_blue,
                scene_bg_color=(1, 1, 1),
            )
        else:
            # Fallback
            renders = []
            for verts, cam_t in zip(all_verts, all_cam_t):
                try:
                    r = self.renderer(
                        verts, cam_t, img_cv2,
                        mesh_base_color=self.config.light_blue,
                        scene_bg_color=(1, 1, 1)
                    )
                    renders.append(r)
                except Exception:
                    renders.append(np.zeros((h_full, w_full, 4), dtype=np.float32))
            
            # Manual alpha compositing
            rendered_combined = np.zeros((h_full, w_full, 4), dtype=np.float32)
            for r in renders:
                alpha = r[..., 3:4]
                if alpha.max() > 1.5: alpha /= 255.0
                rendered_combined[..., :3] = (
                    rendered_combined[..., :3] * (1 - alpha) + r[..., :3] * alpha
                )
                rendered_combined[..., 3:4] = np.clip(rendered_combined[..., 3:4] + alpha, 0, 1)

        # Composite
        if rendered_combined.shape[:2] != (h_full, w_full):
             rendered_combined = cv2.resize(
                 rendered_combined, (w_full, h_full), interpolation=cv2.INTER_LINEAR
             )

        full_img_rgb = img_cv2.astype(np.float32)[:, :, ::-1] / 255.0
        alpha = rendered_combined[..., 3:4].astype(np.float32)
        if alpha.max() > 1.5:
             alpha /= 255.0
        
        rgb_overlay = rendered_combined[..., :3].astype(np.float32)
        final_overlay = full_img_rgb * (1 - alpha) + rgb_overlay * alpha
        final_overlay = np.clip(final_overlay, 0.0, 1.0)
        final_render_bgr = (final_overlay[:, :, ::-1] * 255).astype(np.uint8)

        _, png_bytes = cv2.imencode(".png", final_render_bgr)
        return base64.b64encode(png_bytes).decode("utf-8")

    def _process_image_np(
        self, 
        image_np: np.ndarray, 
        render: bool = True
    ) -> Dict[str, Any]:
        """Internal pipeline: Detect -> Inference -> Projection -> (Optional) Render."""
        img_cv2 = image_np.copy()
        img_rgb = img_cv2[:, :, ::-1]

        # 1. Detect Humans
        pred_bboxes, pred_scores = self._detect_humans(img_cv2)
        if pred_bboxes is None:
            return {"error": "No person detected"}

        # 2. Detect Hands
        bboxes, is_right = self._extract_hand_bboxes(img_rgb, pred_bboxes, pred_scores)
        if not bboxes:
            return {"error": "No hands detected"}

        # 3. Run HaMeR
        boxes_np = np.stack(bboxes)
        right_np = np.stack(is_right)
        inference_results = self._run_hamer_inference(img_cv2, boxes_np, right_np)

        # 4. Process Results (Project to pixels)
        output_data = {"hands": []}
        img_res = (img_cv2.shape[1], img_cv2.shape[0])

        for res in inference_results:
            # Optimized GPU Projections
            verts_px_tensor = self._project_vertices_to_pixels(
                vertices=res["vertices_cuda"],
                cam_t=res["cam_t_cuda"],
                focal_length=res["focal_length"],
                img_res=img_res
            )
            
            verts_planar_tensor, scale_scalar = self._project_vertices_to_planar_z0(
                vertices=res["vertices_cuda"],
                cam_t=res["cam_t_cuda"],
                focal_length=res["focal_length"]
            )
            
            # --- Serialization (Move to CPU now) ---
            verts_3d_list = res["vertices_cuda"].detach().cpu().numpy().tolist()
            verts_px_list = verts_px_tensor.detach().cpu().numpy().tolist()
            verts_planar_list = verts_planar_tensor.detach().cpu().numpy().tolist()
            cam_t_list = res["cam_t_cuda"].detach().cpu().numpy().tolist()

            is_right_hand = bool(res["is_right"])
            faces_arr = self.renderer.faces if is_right_hand else self.renderer.faces_left

            output_data["hands"].append({
                "is_right": is_right_hand,
                "vertices_3d": verts_3d_list,
                "vertices_pixel": verts_px_list,
                "camera_translation": cam_t_list,
                "vertices_planar_z0": verts_planar_list,
                "scaling_factor_planar_z0": scale_scalar,
                "faces": faces_arr.tolist()
            })

        # 5. Render (Optional)
        if render:
            b64_image = self._render_scene(img_cv2, inference_results)
            output_data["full_frame_rendered_image_base64"] = b64_image

        return output_data

    def _process_video(self, file_path: str, sample_rate: int = 100) -> Dict[str, Any]:
        """
        Process video file by sampling frames.
        """
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {file_path}")

        frame_count = 0
        sampled_results = []

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                if frame_count == 1:
                    result = self._process_image_np(frame, render=True)
                    sampled_results.append(result)
                elif frame_count % sample_rate == 0:
                    try:
                        result = self._process_image_np(frame, render=False)
                        sampled_results.append(result)
                    except Exception as e:
                        sampled_results.append({"error": str(e)})

        finally:
            cap.release()

        return {
            "frames_processed": len(sampled_results),
            "samples": sampled_results
        }

    # -------------------------
    # Flask interface methods
    # -------------------------
    
    def process_image_file(self, file) -> Dict[str, Any]:
        """Process an uploaded image file-like object."""
        filestr = file.read()
        npimg = np.frombuffer(filestr, np.uint8)
        image_np = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        return self._process_image_np(image_np, render=True)

    def process_video_file(self, file) -> Dict[str, Any]:
        """Process an uploaded video file-like object."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        try:
            result = self._process_video(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        return result