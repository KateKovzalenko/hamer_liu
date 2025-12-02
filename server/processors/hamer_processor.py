import os
import tempfile
import urllib.request
import base64
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional, Union

import torch
import cv2
import numpy as np

# Third-party library imports
from IPython import embed
from hamer.configs import CACHE_DIR_HAMER
from hamer.models import download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.datasets.vitdet_dataset import ViTDetDataset
from hamer.utils import recursive_to
from hamer.utils.renderer import Renderer, cam_crop_to_full
from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
from detectron2.config import LazyConfig
import hamer

# Local/Custom imports
from ..utils_debug import save_img
from .base_processor import BaseProcessor
from vitpose_model import ViTPoseModel

# --- Constants & Configuration ---
FINAL_MODEL_FILENAME = "model_final_f05665.pkl"
FINAL_MODEL_PATH = str(Path(CACHE_DIR_HAMER) / FINAL_MODEL_FILENAME)
MODEL_URL = (
    "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/"
    "cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
)

ASSETS = [DEFAULT_CHECKPOINT, FINAL_MODEL_PATH]

# --- Global Helper Functions ---

def _assets_ready(cache_dir: Path) -> bool:
    sentinel = cache_dir / ".assets_ready"
    if sentinel.exists():
        return True
    if not all(Path(asset).exists() for asset in ASSETS):
        return False
    sentinel.touch()
    return True

def _remove_tar_gz_files(cache_dir: Union[str, Path]) -> int:
    """Remove top-level .tar.gz files in cache_dir (non-recursive)."""
    cache_path = Path(cache_dir)
    if not cache_path.is_dir():
        return 0
    removed = 0
    for p in cache_path.iterdir():
        if p.is_file() and p.name.endswith(".tar.gz"):
            try:
                p.unlink()
                removed += 1
            except Exception:
                pass
    return removed

def ensure_hamer_assets() -> None:
    cache_dir = Path(CACHE_DIR_HAMER)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    if _assets_ready(cache_dir):
        return

    print("Downloading final model for ViTDet human detector...")
    urllib.request.urlretrieve(MODEL_URL, FINAL_MODEL_PATH)

    print("Downloading HaMeR model and assets...")
    download_models(str(cache_dir))
    
    _assets_ready(cache_dir)
    _remove_tar_gz_files(cache_dir)

# --- Main Class ---

class HamerProcessor(BaseProcessor):
    """Processor implementation using HaMeR 3D Hand Mesh Reconstruction."""

    # Configuration Constants
    CONFIDENCE_THRESHOLD = 0.5
    BOX_THRESHOLD = 0.25
    HAND_KEYPOINT_THRESHOLD = 0.5
    MIN_VALID_KEYPOINTS = 3
    BATCH_SIZE = 8
    LIGHT_BLUE = (0.65, 0.74, 0.86)

    def __init__(self, device: str = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ensure_hamer_assets()
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
        detectron2_cfg.train.init_checkpoint = FINAL_MODEL_PATH
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = self.BOX_THRESHOLD
        self.detector = DefaultPredictor_Lazy(detectron2_cfg)

        # 3. ViTPose (Keypoint Detector)
        self.vitpose = ViTPoseModel(self.device)

        # 4. Renderer
        # We initialize the renderer with the base MANO faces. 
        # The renderer class handles adding watertight faces internally.
        self.renderer = Renderer(self.model_cfg, faces=self.model.mano.faces)

    # --------------------------------------------------------------------------
    # Core Pipeline Steps (SOC: Separation of Concerns)
    # --------------------------------------------------------------------------

    def _detect_humans(self, img_cv2: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Run Detectron2 to find humans."""
        det_out = self.detector(img_cv2)
        det_instances = det_out["instances"]
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > self.CONFIDENCE_THRESHOLD)
        
        if valid_idx.sum() == 0:
            return None, None

        pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores = det_instances.scores[valid_idx].cpu().numpy()
        return pred_bboxes, pred_scores

    def _extract_hand_bboxes(self, img_rgb: np.ndarray, pred_bboxes: np.ndarray, pred_scores: np.ndarray) -> Tuple[List[Any], List[int]]:
        """Run ViTPose and extract hand bounding boxes from body keypoints."""
        vitposes_out = self.vitpose.predict_pose(
            img_rgb, 
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)]
        )

        bboxes = []
        is_right = []

        for vitposes in vitposes_out:
            # Indices for hands in common pose formats (usually COCO-like)
            left_hand_keyp = vitposes["keypoints"][-42:-21]
            right_hand_keyp = vitposes["keypoints"][-21:]

            for keypoints, is_r_flag in [(left_hand_keyp, 0), (right_hand_keyp, 1)]:
                valid = keypoints[:, 2] > self.HAND_KEYPOINT_THRESHOLD
                if valid.sum() > self.MIN_VALID_KEYPOINTS:
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
            dataset, batch_size=self.BATCH_SIZE, shuffle=False, num_workers=0
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
            scaled_focal_length = (
                self.model_cfg.EXTRA.FOCAL_LENGTH / 
                self.model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            )

            pred_cam_t_full = cam_crop_to_full(
                pred_cam, box_center, box_size, img_size, scaled_focal_length
            ).detach().cpu().numpy()

            batch_len = pred_cam.shape[0]
            for n in range(batch_len):
                verts_n = out["pred_vertices"][n]  # Keep on CUDA for rendering later if needed
                is_r_n = int(batch["right"][n].detach().item())
                
                # Flip left hand vertices to match right hand canonical space if needed
                # (Logic copied from original: Left hand needs X-flip)
                verts_n[:, 0] = (2 * is_r_n - 1) * verts_n[:, 0]
                
                cam_t_n = pred_cam_t_full[n]
                
                inference_results.append({
                    "vertices_cuda": verts_n,
                    "cam_t": cam_t_n,
                    "is_right": is_r_n,
                    "focal_length": scaled_focal_length
                })
        
        return inference_results

    def _project_vertices_to_pixels(
            self,
            vertices: torch.Tensor,
            cam_t: np.ndarray,
            focal_length: float,
            img_res: tuple[int, int]
        ) -> Tuple[np.ndarray, np.ndarray]:
            """
            Convert 3D verts to Camera Space XYZ and Pixel Space XY.
            Matches the transformation logic in renderer.py -> render_rgba_multiple.
            """
            
            # 1. Prepare Inputs
            if isinstance(focal_length, torch.Tensor):
                focal_length = float(focal_length.detach().cpu())

            verts_cpu = vertices.detach().cpu().numpy()
            tx, ty, tz = cam_t

            # 2. Apply Camera Translation (World -> Camera Frame Base)
            # Matches `vertices_to_trimesh`: mesh = Trimesh(vertices + camera_translation, ...)
            X_trans = verts_cpu[:, 0] + tx
            Y_trans = verts_cpu[:, 1] + ty
            Z_trans = verts_cpu[:, 2] + tz

            # 3. Apply Rotation (RotX 180 degrees)
            # Matches `vertices_to_trimesh`: mesh.apply_transform(rot_matrix(180, [1,0,0]))
            # RotX(180) maps: (x, y, z) -> (x, -y, -z)
            X_view = X_trans
            Y_view = Y_trans
            Z_view = -Z_trans

            # 4. Perspective Projection
            # PyRender/OpenGL Camera looks down -Z axis.
            # Depth is distance along the Z axis: depth = -Z_view
            # Matches `pyrender.IntrinsicsCamera` projection logic.
            depth = -Z_view + 1e-8 # Avoid division by zero
            
            W, H = img_res
            cx, cy = W / 2.0, H / 2.0

            # Standard Pinhole Projection: u = fx * (x / depth) + cx
            u = focal_length * (X_view / depth) + cx
            v = focal_length * (Y_view / depth) + cy

            # 5. Pack Results
            # We return the "Camera Space" coordinates as (X_view, Y_view, depth)
            # so that Z represents positive depth from the camera.
            verts_cam = np.stack([X_view, Y_view, depth], axis=1)
            verts_px = np.stack([u, v], axis=1)

            return verts_cam, verts_px

    def _project_vertices_to_planar_z0(
        self,
        vertices: torch.Tensor,
        cam_t: np.ndarray,
        focal_length: float
    ) -> np.ndarray:
        """
        New Method: Project 3D vertices onto the Z=0 plane while preserving 
        perspective-based scaling and relative volume.
        
        This shifts all hands to have a mean Z of 0, but scales the Z values 
        so that the hand's volume (aspect ratio) remains consistent with the 
        projected X and Y dimensions.
        """
        if isinstance(focal_length, torch.Tensor):
            focal_length = float(focal_length.detach().cpu())

        verts_cpu = vertices.detach().cpu().numpy()
        tx, ty, tz = cam_t

        # 1. Transform to Camera Space (preserve relative spatial coords)
        X = verts_cpu[:, 0] + tx
        Y = (verts_cpu[:, 1] + ty)
        Z = verts_cpu[:, 2] + tz + 1e-8  # Avoid division by zero

        # 2. Calculate Perspective Scale Factor for X and Y
        # This determines how much to shrink/expand based on distance from camera.
        scale_factor = focal_length / Z

        # 3. Apply Projection to X and Y (matches image perspective)
        X_planar = X * scale_factor
        Y_planar = Y * scale_factor
        
        # 4. Handle Z (Volume Preservation)
        # To keep the "thickness" of the hand proportional to the projected X/Y,
        # we scale the Z values by the 'weak perspective' scale factor (scale at the centroid).
        # We then shift the result so the mean Z is 0.
        Z_mean = np.mean(Z)
        scale_mean = focal_length / Z_mean
        
        # Z_planar centers the volume at 0 and scales it to match the view scale
        Z_planar = (Z - Z_mean) * scale_mean

        return np.stack([X_planar, Y_planar, Z_planar], axis=1)

    def _render_scene(
        self, 
        img_cv2: np.ndarray, 
        inference_results: List[Dict[str, Any]]
    ) -> str:
        """Render mesh overlay on the image and return Base64 string."""
        if not inference_results:
            return ""

        all_verts = [res["vertices_cuda"].detach().cpu().numpy() for res in inference_results]
        all_cam_t = [res["cam_t"] for res in inference_results]
        all_right = [res["is_right"] for res in inference_results]

        h_full, w_full = img_cv2.shape[:2]
        
        # Attempt to use efficient batch renderer if available
        render_fn = getattr(self.renderer, "render_rgba_multiple", None)
        
        # Calculate focal length based on first result (assuming uniform image batch)
        # Note: Original code re-calculates this per batch, but usually it's per image.
        # We take the first valid one.
        focal_length = float(inference_results[0]["focal_length"])
        # Original logic re-scales focal length for full render resolution
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
                mesh_base_color=self.LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
            )
        else:
            # Fallback manual rendering loop
            renders = []
            for verts, cam_t in zip(all_verts, all_cam_t):
                try:
                    r = self.renderer(
                        verts, cam_t, img_cv2,
                        mesh_base_color=self.LIGHT_BLUE,
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

        # Composite over original image
        # Resize if necessary (safety check)
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

    # --------------------------------------------------------------------------
    # Main Processing Logic (Clean & Linear)
    # --------------------------------------------------------------------------

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
            # Standard Projection
            verts_cam, verts_px = self._project_vertices_to_pixels(
                vertices=res["vertices_cuda"],
                cam_t=res["cam_t"],
                focal_length=res["focal_length"],
                img_res=img_res
            )
            
            # New Planar Projection (Z centered at 0 with volume scaling)
            verts_planar = self._project_vertices_to_planar_z0(
                vertices=res["vertices_cuda"],
                cam_t=res["cam_t"],
                focal_length=res["focal_length"]
            )
            
            # CRITICAL FIX: Convert Tensor to list for JSON serialization
            # res["vertices_cuda"] is a Torch Tensor on GPU/CPU.
            verts_3d_list = res["vertices_cuda"].detach().cpu().numpy().tolist()

            # Retrieve the correct faces from the renderer to ensure the mesh is 
            # consistent with the rendering (watertight, correct winding order).
            is_right_hand = bool(res["is_right"])
            faces_arr = self.renderer.faces if is_right_hand else self.renderer.faces_left

            output_data["hands"].append({
                "is_right": is_right_hand,
                "vertices_3d": verts_3d_list,
                "vertices_pixel": verts_px.tolist(),
                "camera_translation": res["cam_t"].tolist(),
                "vertices_planar_z0": verts_planar.tolist(),
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
        Uses the same pipeline as image processing to ensure DRY.
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
                
                # Logic: Process first frame fully (with render), 
                # sample subsequent frames without render.
                if frame_count == 1:
                    result = self._process_image_np(frame, render=True)
                    sampled_results.append(result)
                elif frame_count % sample_rate == 0:
                    # Reuse the same pipeline, just disable rendering
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
        """Process an uploaded image file-like object (from Flask)."""
        filestr = file.read()
        npimg = np.frombuffer(filestr, np.uint8)
        image_np = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        return self._process_image_np(image_np, render=True)

    def process_video_file(self, file) -> Dict[str, Any]:
        """Process an uploaded video file-like object (from Flask)."""
        # Save uploaded file to a temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        try:
            result = self._process_video(tmp_path)
        finally:
            # Ensure cleanup happens even if processing crashes
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        return result