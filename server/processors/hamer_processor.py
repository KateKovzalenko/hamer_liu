import os
import tempfile
import torch
import cv2
import numpy as np
import base64
from pathlib import Path
from typing import Dict, Any

from .base_processor import BaseProcessor

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.datasets.vitdet_dataset import ViTDetDataset
from hamer.utils import recursive_to
from hamer.utils.renderer import Renderer, cam_crop_to_full
from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy

from vitpose_model import ViTPoseModel


class HamerProcessor(BaseProcessor):
    """Processor implementation using HaMeR 3D Hand Mesh Reconstruction."""

    def __init__(self, device: str = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # Download & load model
        download_models(CACHE_DIR_HAMER)
            
        self.model, self.model_cfg = load_hamer(DEFAULT_CHECKPOINT)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Load human detector (ViTDet)
        from detectron2.config import LazyConfig
        import hamer
        cfg_path = Path(hamer.__file__).parent / "configs" / "cascade_mask_rcnn_vitdet_h_75ep.py"
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = (
            "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/"
            "cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        )
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        self.detector = DefaultPredictor_Lazy(detectron2_cfg)

        # Load keypoint detector
        self.vitpose = ViTPoseModel(self.device)

        # Renderer setup
        self.renderer = Renderer(self.model_cfg, faces=self.model.mano.faces)
        self.LIGHT_BLUE = (0.65, 0.74, 0.86)

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

            # --- Step 4: Render result ---
            rendered = self.renderer(
                verts,
                out["pred_cam_t"][0].detach().cpu().numpy(),
                batch["img"][0],
                mesh_base_color=self.LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
            )
            rendered_bgr = (rendered[:, :, ::-1] * 255).astype(np.uint8)

            # Convert to Base64 for JSON
            _, png_bytes = cv2.imencode(".png", rendered_bgr)
            rendered_b64 = base64.b64encode(png_bytes).decode("utf-8")

            all_results["hands"].append(
                {
                    "is_right": is_r,
                    "vertices": verts.tolist(),
                    "camera_translation": cam_t.tolist(),
                    "rendered_image_base64": rendered_b64,
                }
            )
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