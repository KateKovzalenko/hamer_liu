import os
import tempfile
import urllib.request
import base64
import contextlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, List, Optional, Union, Literal

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

# Assuming these exist in your environment based on original code
from .base_processor import BaseProcessor
from vitpose_model import ViTPoseModel

# ------------------------------------------------------------------------------
# 1. Configuration & Constants
# ------------------------------------------------------------------------------

@dataclass(frozen=True)
class HamerConfig:
    """
    Immutable configuration for the Hamer Pipeline.
    """
    device: str
    confidence_threshold: float = 0.5
    box_threshold: float = 0.25
    hand_keypoint_threshold: float = 0.5
    min_valid_keypoints: int = 3
    batch_size: int = 8
    light_blue: Tuple[float, float, float] = (0.65, 0.74, 0.86)
    
    # Asset Management
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


# ------------------------------------------------------------------------------
# 2. Asset Management (Separated from Runtime)
# ------------------------------------------------------------------------------

class AssetManager:
    """
    Responsible for ensuring model weights and assets exist.
    """
    @staticmethod
    def ensure_assets(config: HamerConfig) -> None:
        config.cache_dir.mkdir(parents=True, exist_ok=True)
        
        sentinel = config.cache_dir / ".assets_ready"
        assets = [DEFAULT_CHECKPOINT, str(config.final_model_path)]
        
        if sentinel.exists() and all(Path(a).exists() for a in assets):
            return

        print("[AssetManager] Downloading final model for ViTDet human detector...")
        if not config.final_model_path.exists():
            urllib.request.urlretrieve(config.model_url, str(config.final_model_path))

        print("[AssetManager] Downloading HaMeR model and assets...")
        download_models(str(config.cache_dir))
        
        AssetManager._cleanup_tarballs(config.cache_dir)
        sentinel.touch()

    @staticmethod
    def _cleanup_tarballs(cache_dir: Path) -> None:
        if not cache_dir.is_dir():
            return
        for p in cache_dir.iterdir():
            if p.is_file() and p.name.endswith(".tar.gz"):
                with contextlib.suppress(OSError):
                    p.unlink()


# ------------------------------------------------------------------------------
# 3. Mathematical Utilities (Pure Logic)
# ------------------------------------------------------------------------------

class ProjectionUtils:
    """
    Stateless geometric projection utilities optimized for GPU tensors.
    """
    @staticmethod
    def project_to_pixels(
        vertices: torch.Tensor,
        cam_t: torch.Tensor,
        focal_length: float,
        img_res: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Standard pinhole projection.
        """
        tx, ty, tz = cam_t[0], cam_t[1], cam_t[2]
        
        # Camera translation
        X_trans = vertices[:, 0] + tx
        Y_trans = vertices[:, 1] + ty
        Z_trans = vertices[:, 2] + tz
        
        # RotX(180) -> (x, -y, -z)
        X_view = X_trans
        Y_view = Y_trans
        Z_view = -Z_trans
        
        depth = -Z_view + 1e-8
        
        W, H = img_res
        cx, cy = W / 2.0, H / 2.0
        
        u = focal_length * (X_view / depth) + cx
        v = focal_length * (Y_view / depth) + cy
        
        return torch.stack([u, v], dim=1)

    @staticmethod
    def project_to_planar_z0(
        vertices: torch.Tensor,
        cam_t: torch.Tensor,
        focal_length: float
    ) -> Tuple[torch.Tensor, float]:
        """
        Project 3D vertices onto Z=0 plane preserving perspective scale.
        """
        tx, ty, tz = cam_t[0], cam_t[1], cam_t[2]
        
        X = vertices[:, 0] + tx
        Y = (vertices[:, 1] + ty) # Note: Y is usually flipped in cam space, treating raw here
        Z = vertices[:, 2] + tz + 1e-8
        
        # Perspective scale factor
        scale_factor_vec = focal_length / Z
        
        X_planar = X * scale_factor_vec
        Y_planar = Y * scale_factor_vec
        
        Z_mean = torch.mean(Z)
        scale_mean = focal_length / Z_mean
        Z_planar = (Z - Z_mean) * scale_mean
        
        return torch.stack([X_planar, Y_planar, Z_planar], dim=1), scale_mean.item()


# ------------------------------------------------------------------------------
# 4. Pipeline Components (Detectors & Model Wrappers)
# ------------------------------------------------------------------------------

class HumanDetector:
    def __init__(self, config: HamerConfig):
        self.config = config
        cfg_path = Path(hamer.__file__).parent / "configs" / "cascade_mask_rcnn_vitdet_h_75ep.py"
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = str(config.final_model_path)
        
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = config.box_threshold
            
        self.predictor = DefaultPredictor_Lazy(detectron2_cfg)

    def detect(self, img_cv2: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        out = self.predictor(img_cv2)
        instances = out["instances"]
        valid_idx = (instances.pred_classes == 0) & (instances.scores > self.config.confidence_threshold)
        
        if valid_idx.sum() == 0:
            return None, None
            
        return (
            instances.pred_boxes.tensor[valid_idx].cpu().numpy(),
            instances.scores[valid_idx].cpu().numpy()
        )

class HamerProcessor(BaseProcessor):
    """
    Facade orchestrating the 3D Hand Reconstruction Pipeline.
    Refactored for maintainability, lazy loading, and architectural separation.
    """

    def __init__(self, device: Optional[str] = None):
        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.config = HamerConfig(device=resolved_device)
        self.device = torch.device(self.config.device)
        
        # Initialize Assets
        AssetManager.ensure_assets(self.config)
        
        # Models are initialized immediately as per original requirement, 
        # though lazy loading would be architecturally superior for startup time.
        self._init_models()

    def _init_models(self):
        """Initialize all neural network models via wrappers or direct loading."""
        # 1. HaMeR Model
        self.hamer_model, self.model_cfg = load_hamer(DEFAULT_CHECKPOINT)
        self.hamer_model = self.hamer_model.to(self.device)
        self.hamer_model.eval()

        # 2. Detectors
        self.human_detector = HumanDetector(self.config)
        self.vitpose = ViTPoseModel(self.device)

        # 3. Renderer
        self.renderer = Renderer(self.model_cfg, faces=self.hamer_model.mano.faces)

    # --------------------------------------------------------------------------
    # Logic: Filtering & Selection
    # --------------------------------------------------------------------------

    def _filter_humans(
        self, 
        bboxes: np.ndarray, 
        scores: np.ndarray, 
        selector: Union[str, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        if selector == "all" or len(bboxes) == 0:
            return bboxes, scores

        # Calculate X-centroids
        centroids_x = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
        sorted_indices = np.argsort(centroids_x)
        num_people = len(bboxes)
        target_idx = 0

        if isinstance(selector, int):
            target_idx = max(0, min(selector, num_people - 1))
        elif isinstance(selector, str):
            sel = selector.lower()
            if sel in ["right", "rightmost"]: target_idx = num_people - 1
            elif sel == "center": target_idx = num_people // 2
            # default "left" is 0
        
        real_idx = sorted_indices[target_idx]
        return bboxes[real_idx:real_idx+1], scores[real_idx:real_idx+1]

    def _extract_hands(
        self,
        img_rgb: np.ndarray,
        person_bboxes: np.ndarray,
        person_scores: np.ndarray,
        side_selector: str
    ) -> List[Dict[str, Any]]:
        """Run ViTPose and extract hand bounding boxes."""
        # Prepare input for ViTPose: [x1, y1, x2, y2, score]
        boxes_with_scores = np.concatenate([person_bboxes, person_scores[:, None]], axis=1)
        vitposes_out = self.vitpose.predict_pose(img_rgb, [boxes_with_scores])
        
        candidates = []
        target = side_selector.lower()
        
        for i, vitposes in enumerate(vitposes_out):
            p_bbox = person_bboxes[i]
            p_score = person_scores[i]
            
            # (keypoints_slice, is_right_flag)
            checks = []
            if target in ["left", "both"]: checks.append((vitposes["keypoints"][-42:-21], 0))
            if target in ["right", "both"]: checks.append((vitposes["keypoints"][-21:], 1))

            for kps, is_right in checks:
                valid_mask = kps[:, 2] > self.config.hand_keypoint_threshold
                if valid_mask.sum() > self.config.min_valid_keypoints:
                    valid_kps = kps[valid_mask]
                    candidates.append({
                        "hand_bbox": [
                            valid_kps[:, 0].min(), valid_kps[:, 1].min(),
                            valid_kps[:, 0].max(), valid_kps[:, 1].max()
                        ],
                        "is_right": is_right,
                        "person_bbox": p_bbox.tolist(),
                        "person_score": float(p_score)
                    })
        return candidates

    # --------------------------------------------------------------------------
    # Logic: Inference Loop
    # --------------------------------------------------------------------------

    def _run_hamer(self, img_cv2: np.ndarray, candidates: List[Dict]) -> List[Dict]:
        """Batched HaMeR inference."""
        boxes = np.array([c["hand_bbox"] for c in candidates])
        right_flags = np.array([c["is_right"] for c in candidates])
        
        dataset = ViTDetDataset(self.model_cfg, img_cv2, boxes, right_flags)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=0
        )
        
        results = []
        for batch in loader:
            batch = recursive_to(batch, self.device)
            with torch.no_grad():
                out = self.hamer_model(batch)

            # Camera logic
            pred_cam = out["pred_cam"]
            multiplier = (2 * batch["right"] - 1)
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]
            
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            
            scaled_focal_length = (
                self.model_cfg.EXTRA.FOCAL_LENGTH / 
                self.model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            )

            pred_cam_full = cam_crop_to_full(
                pred_cam, box_center, box_size, img_size, scaled_focal_length
            )

            for n in range(pred_cam.shape[0]):
                verts = out["pred_vertices"][n]
                is_r = int(batch["right"][n].item())
                
                # Normalize Left hand to Right hand canonical space if needed
                if is_r == 0:
                    verts[:, 0] = -1 * verts[:, 0]

                results.append({
                    "vertices": verts, # GPU
                    "cam_t": pred_cam_full[n], # GPU
                    "is_right": is_r,
                    "focal_length": scaled_focal_length # Scalar Tensor
                })
        return results

    # --------------------------------------------------------------------------
    # Logic: Serialization & Rendering
    # --------------------------------------------------------------------------

    def _serialize_results(
        self, 
        inference_results: List[Dict], 
        candidates: List[Dict],
        img_res: Tuple[int, int]
    ) -> List[Dict]:
        serialized_hands = []
        
        for res, candidate in zip(inference_results, candidates):
            focal = res["focal_length"].item() if isinstance(res["focal_length"], torch.Tensor) else res["focal_length"]
            
            # GPU Math
            px_coords = ProjectionUtils.project_to_pixels(
                res["vertices"], res["cam_t"], focal, img_res
            )
            planar_coords, scale = ProjectionUtils.project_to_planar_z0(
                res["vertices"], res["cam_t"], focal
            )
            
            # To CPU
            is_right = bool(res["is_right"])
            faces = self.renderer.faces if is_right else self.renderer.faces_left
            
            serialized_hands.append({
                "is_right": is_right,
                "vertices_3d": res["vertices"].detach().cpu().numpy().tolist(),
                "vertices_pixel": px_coords.detach().cpu().numpy().tolist(),
                "camera_translation": res["cam_t"].detach().cpu().numpy().tolist(),
                "vertices_planar_z0": planar_coords.detach().cpu().numpy().tolist(),
                "scaling_factor_planar_z0": scale,
                "faces": faces.tolist(),
                "person_bounding_box_xyxy": candidate["person_bbox"],
                "person_detection_score": candidate["person_score"]
            })
            
        return serialized_hands

    def _render_overlay(self, img_cv2: np.ndarray, results: List[Dict]) -> str:
        """Renders scene and returns base64 PNG."""
        if not results: return ""
        
        h, w = img_cv2.shape[:2]
        
        # Prepare data for renderer (CPU/Numpy expected by HaMeR renderer)
        verts_list = [r["vertices"].detach().cpu().numpy() for r in results]
        cam_list = [r["cam_t"].detach().cpu().numpy() for r in results]
        right_list = [r["is_right"] for r in results]
        
        # Determine focal length (assumes uniform batch)
        focal = results[0]["focal_length"]
        if isinstance(focal, torch.Tensor): focal = focal.item()
        
        scaled_focal = (self.model_cfg.EXTRA.FOCAL_LENGTH / 
                        self.model_cfg.MODEL.IMAGE_SIZE * max(h, w))

        # Render
        # Ideally, we use render_rgba_multiple if available for batching
        render_fn = getattr(self.renderer, "render_rgba_multiple", None)
        
        if render_fn:
            rgba = render_fn(
                verts_list, cam_t=cam_list, render_res=(w, h), is_right=right_list,
                focal_length=scaled_focal, mesh_base_color=self.config.light_blue,
                scene_bg_color=(1, 1, 1)
            )
        else:
            # Fallback sequential render
            layers = []
            for v, c, r in zip(verts_list, cam_list, right_list):
                 # Temporarily swap faces for left hand if needed (renderer usually handles logic via arg but explicit is safer)
                 # Note: renderer.faces is usually static. We rely on the renderer detecting 'is_right' logic 
                 # or we assume vertices are already canonical.
                 # HaMeR standard renderer needs manual handling? 
                 # Assuming Hamer renderer handles mirroring via 'is_right' flag internally or we just pass canonical verts
                 layers.append(self.renderer(v, c, img_cv2, mesh_base_color=self.config.light_blue))
            
            # Alpha composite manually
            rgba = np.zeros((h, w, 4), dtype=np.float32)
            for layer in layers:
                alpha = layer[..., 3:4] / 255.0 if layer.max() > 1.5 else layer[..., 3:4]
                rgba[..., :3] = rgba[..., :3] * (1 - alpha) + layer[..., :3] * alpha
                rgba[..., 3:4] = np.clip(rgba[..., 3:4] + alpha, 0, 1)

        # Final Composite
        if rgba.shape[:2] != (h, w):
            rgba = cv2.resize(rgba, (w, h))

        img_float = img_cv2.astype(np.float32)[:, :, ::-1] / 255.0
        alpha_o = rgba[..., 3:4]
        if alpha_o.max() > 1.5: alpha_o /= 255.0
        
        comp = img_float * (1 - alpha_o) + rgba[..., :3] * alpha_o
        comp_bgr = (np.clip(comp, 0, 1) * 255).astype(np.uint8)[:, :, ::-1]
        
        return base64.b64encode(cv2.imencode(".png", comp_bgr)[1]).decode("utf-8")

    # --------------------------------------------------------------------------
    # Public Entry Points (Internal)
    # --------------------------------------------------------------------------

    def _process_frame_logic(
        self, 
        image_np: np.ndarray, 
        render: bool,
        person_selector: Union[str, int],
        hand_side: str
    ) -> Dict[str, Any]:
        """Internal pipeline execution."""
        img_rgb = image_np[:, :, ::-1].copy() # ViTPose needs RGB
        
        # 1. Detect People
        p_boxes, p_scores = self.human_detector.detect(image_np)
        if p_boxes is None: return {"error": "No person detected"}
        
        # 2. Filter People
        p_boxes, p_scores = self._filter_humans(p_boxes, p_scores, person_selector)
        if len(p_boxes) == 0: return {"error": "Person selection failed"}
        
        # 3. Extract Hands
        hand_candidates = self._extract_hands(img_rgb, p_boxes, p_scores, hand_side)
        if not hand_candidates: return {"error": "No hands detected"}
        
        # 4. Mesh Inference
        inference_results = self._run_hamer(image_np, hand_candidates)
        
        # 5. Output Generation
        output = {
            "hands": self._serialize_results(inference_results, hand_candidates, (image_np.shape[1], image_np.shape[0]))
        }
        
        if render:
            output["full_frame_rendered_image_base64"] = self._render_overlay(image_np, inference_results)
            
        return output

    # --------------------------------------------------------------------------
    # Flask Interface Methods (Preserved Signature)
    # --------------------------------------------------------------------------

    def process_image_file(
        self, 
        file, 
        person_selector: Union[str, int] = "all", 
        hand_side: str = "both"
    ) -> Dict[str, Any]:
        """
        Process an uploaded image file-like object.
        """
        try:
            filestr = file.read()
            npimg = np.frombuffer(filestr, np.uint8)
            image_np = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            if image_np is None:
                return {"error": "Failed to decode image"}
                
            return self._process_frame_logic(image_np, True, person_selector, hand_side)
        except Exception as e:
            # Log error in production
            return {"error": f"Processing failure: {str(e)}"}

    def process_video_file(
        self, 
        file, 
        person_selector: Union[str, int] = "all", 
        hand_side: str = "both"
    ) -> Dict[str, Any]:
        """
        Process an uploaded video file-like object.
        """
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        results = []
        frame_idx = 0
        sample_rate = 100 # Could be parameterized in config
        
        try:
            if not cap.isOpened():
                return {"error": "Could not open video file"}

            while True:
                ret, frame = cap.read()
                if not ret: break
                
                frame_idx += 1
                should_render = (frame_idx == 1)
                should_process = (frame_idx == 1) or (frame_idx % sample_rate == 0)
                
                if should_process:
                    try:
                        res = self._process_frame_logic(frame, should_render, person_selector, hand_side)
                        results.append(res)
                    except Exception as e:
                        results.append({"frame": frame_idx, "error": str(e)})
                        
        finally:
            cap.release()
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        return {
            "frames_processed": len(results),
            "samples": results
        }