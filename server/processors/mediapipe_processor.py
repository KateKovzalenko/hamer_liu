import os
import cv2
import numpy as np
import mediapipe as mp
import tempfile
from .base_processor import BaseProcessor


class MediapipeProcessor(BaseProcessor):
    """Processor implementation using MediaPipe Hands."""

    def __init__(self):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5
        )
        self.drawing = mp.solutions.drawing_utils

    # -------------------------
    # Internal helper
    # -------------------------
    def _process(self, image_np):
        """Internal helper to process a NumPy image array."""
        results = self.hands.process(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
        response = {"vertices": [], "confidence_scores": []}

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    response["vertices"].append({"x": lm.x, "y": lm.y, "z": lm.z})
                    response["confidence_scores"].append(0.95)
        return response
        
    def _process_video(self, file_path):
        """Process video file by sampling every Nth frame."""
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {file_path}")

        frame_count = 0
        sampled_results = []
        sample_rate = 10  # process every 10th frame

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            if frame_count % sample_rate == 0:
                sampled_results.append(self._process(frame))

        cap.release()
        return {
            "frames_processed": len(sampled_results),
            "samples": sampled_results
        }


    # -------------------------
    # Public methods
    # -------------------------
    def process_image_file(self, file):
        """Process an uploaded image file-like object."""
        filestr = file.read()
        npimg = np.frombuffer(filestr, np.uint8)
        image_np = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        return self._process(image_np)

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
    