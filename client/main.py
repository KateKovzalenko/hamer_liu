import argparse
import base64
import json
import logging
import os
import sys
import tkinter as tk
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from tkinter import filedialog
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons

import requests
from PIL import Image

# Assumed existing modules per prompt instructions
try:
    from client_api import ClientAPI
    import results_handler
    import visualization
except ImportError:
    # Fallback for standalone analysis context if local modules are missing
    logging.warning("Local modules (client_api, results_handler, visualization) not found. Mocking for structural validity.")
    class ClientAPI:
        def __init__(self, url): pass
        def upload_image(self, p): return {}
        def upload_video(self, p): return {}
    class visualization:
        @staticmethod
        def show_image_result(*args, **kwargs): pass
    class results_handler:
        @staticmethod
        def save_json_output(*args): return "timestamp"
        @staticmethod
        def save_video_response(*args): pass
        @staticmethod
        def save_image_response(*args): pass

# -----------------------------------------------------
# Logging Configuration
# -----------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------
# Configuration
# -----------------------------------------------------
@dataclass
class AppConfig:
    host: str = "localhost"
    port: str = "8080"
    output_dir: Path = Path("demo_out")
    default_input_dir: Path = Path("example_data")

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

# -----------------------------------------------------
# Service: Input Management
# -----------------------------------------------------
class InputManager:
    """Handles CLI arguments and GUI file selection."""
    
    @staticmethod
    def get_input_path(cli_path: Optional[str], start_dir: Path) -> Optional[Path]:
        if cli_path:
            return Path(cli_path)

        logger.info("No --input provided, opening GUI file dialog...")
        return InputManager._open_file_dialog(start_dir)

    @staticmethod
    def _open_file_dialog(start_dir: Path) -> Optional[Path]:
        # Initialize Tkinter specifically for this operation
        root = tk.Tk()
        root.withdraw()

        initial_dir = start_dir if start_dir.is_dir() else Path.cwd()
        
        filetypes = [
            ("Media Files", ".jpg .jpeg .png .mp4 .avi .mov .mkv"),
            ("Image Files", ".jpg .jpeg .png"),
            ("Video Files", ".mp4 .avi .mov .mkv"),
            ("All Files", "*.*")
        ]

        file_path_str = filedialog.askopenfilename(
            initialdir=str(initial_dir),
            title="Select Image or Video File",
            filetypes=filetypes
        )
        
        root.destroy() # Ensure cleanup
        
        if not file_path_str:
            logger.warning("No file selected.")
            return None
            
        return Path(file_path_str)

# -----------------------------------------------------
# Service: Data Extraction logic
# -----------------------------------------------------
class DataExtractor:
    """
    Parses raw API JSON responses to extract specific artifacts 
    (images, vertices) regardless of input type (Video/Image).
    """

    @staticmethod
    def decode_base64_image(b64_string: str) -> Optional[Image.Image]:
        try:
            img_data = base64.b64decode(b64_string)
            return Image.open(BytesIO(img_data))
        except Exception as e:
            logger.error(f"Could not decode base64 string: {e}")
            return None

    @staticmethod
    def get_hand_entries(data: Dict[str, Any], is_video: bool) -> List[Dict[str, Any]]:
        """
        Normalizes the response structure. 
        Returns a list of 'hand' dictionaries from the relevant frame(s).
        """
        if not is_video:
            # Single frame response
            return data.get("hands", [])
        
        # Video response: Use the first sample that actually contains hands
        samples = data.get("samples", [])
        for sample in samples:
            if sample and sample.get("hands"):
                return sample["hands"]
        return []

    @classmethod
    def extract_images(cls, data: Dict[str, Any], is_video: bool) -> Tuple[List[Image.Image], str]:
        """Returns list of PIL Images and a display title."""
        images = []
        title = "Render"

        # Priority 1: Full Frame Render (Common to both)
        full_b64 = data.get("full_frame_rendered_image_base64")
        if full_b64:
            img = cls.decode_base64_image(full_b64)
            if img:
                return [img], "HAMER Full Frame"

        # Priority 2: Per-hand crops (Normalized logic)
        hands = cls.get_hand_entries(data, is_video)
        for hand in hands:
            b64 = hand.get("rendered_image_base64")
            if b64:
                img = cls.decode_base64_image(b64)
                if img:
                    images.append(img)

        if images:
            title = "HAMER Hands" if not is_video else "First frame render"
        
        return images, title

    @classmethod
    def extract_vertices(cls, data: Dict[str, Any], is_video: bool, key: str = "vertices_3d", apply_translation: bool = False) -> List[List[List[float]]]:
        """
        Extracts point clouds (3D or 2D).
        
        Args:
            data: The JSON response data.
            is_video: Boolean flag for video mode.
            key: The key in the hand dictionary to extract (e.g., 'vertices_3d', 'vertices_pixel').
            apply_translation: Whether to apply camera translation logic (only for 3D).
        """
        point_clouds = []
        hands = cls.get_hand_entries(data, is_video)

        if is_video and not hands:
            logger.info("No frames with hand data found in video response.")

        logger.info(f"Processing {len(hands)} detected hands for vertex extraction ({key}).")

        for i, hand in enumerate(hands):
            vertices = hand.get(key)
            if not vertices:
                logger.warning(f"Hand {i}: No '{key}' key found.")
                continue

            translation = hand.get('camera_translation')
            
            if apply_translation and translation:
                vertices = cls._apply_translation_math(vertices, translation, i)
            
            point_clouds.append(vertices)

        return point_clouds

    @staticmethod
    def _apply_translation_math(vertices: List[List[float]], translation: List[float], hand_index: int) -> List[List[float]]:
        """Applies camera translation vector to vertex list."""
        if not (isinstance(translation, (list, tuple)) and len(translation) == 3):
            logger.warning(f"Hand {hand_index}: Invalid 'camera_translation'. Using original vertices.")
            return vertices

        try:
            tx, ty, tz = translation
            # Logic preserved from original: tz is divided by 10
            tz = tz / 10.0 
            return [
                [v[0] + tx, v[1] + ty, v[2] + tz] 
                for v in vertices
            ]
        except (TypeError, IndexError) as e:
            logger.error(f"Hand {hand_index}: Math error applying translation: {e}")
            return vertices

# -----------------------------------------------------
# Service: Interactive Visualization
# -----------------------------------------------------
class InteractivePlotter:
    """Handles 2D/3D visualization with interactive UI controls."""
    
    @staticmethod
    def plot_3d_vertices(point_clouds: List[List[List[float]]], title: str, block: bool = False):
        """
        Plots multiple hands in 3D with checkboxes to toggle visibility.
        """
        if not point_clouds:
            logger.warning(f"No data to plot for {title}.")
            return

        fig = plt.figure(figsize=(10, 7))
        plt.subplots_adjust(left=0.25)
        
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(title)
        
        plots = []
        labels = []
        colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black']
        
        for i, vertices in enumerate(point_clouds):
            xs = [v[0] for v in vertices]
            ys = [v[1] for v in vertices]
            zs = [v[2] for v in vertices]
            
            color = colors[i % len(colors)]
            label = f"Hand {i}"
            
            scatter = ax.scatter(xs, ys, zs, c=color, marker='o', s=10, label=label)
            plots.append(scatter)
            labels.append(label)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        InteractivePlotter._setup_checkboxes(fig, plots, labels)
        
        logger.info(f"Opening interactive 3D plot: {title}")
        plt.show(block=block)

    @staticmethod
    def plot_2d_vertices(point_clouds: List[List[List[float]]], title: str, bg_image: Optional[Image.Image] = None, block: bool = False):
        """
        Plots multiple hands in 2D (Pixel Plane) with checkboxes.
        If bg_image is provided, it is plotted as the background.
        """
        if not point_clouds:
            logger.warning(f"No data to plot for {title}.")
            return

        fig = plt.figure(figsize=(10, 7))
        plt.subplots_adjust(left=0.25)
        
        ax = fig.add_subplot(111)
        ax.set_title(title)

        # Plot background image if available
        if bg_image:
            # imshow handles the axis limits and orientation (0,0 at top left) automatically
            ax.imshow(bg_image)
        
        plots = []
        labels = []
        colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black']
        
        for i, vertices in enumerate(point_clouds):
            xs = [v[0] for v in vertices]
            ys = [v[1] for v in vertices]
            
            color = colors[i % len(colors)]
            label = f"Hand {i}"
            
            # s=10 is point size
            scatter = ax.scatter(xs, ys, c=color, marker='o', s=5, label=label)
            plots.append(scatter)
            labels.append(label)

        ax.set_xlabel('X (Pixels)')
        ax.set_ylabel('Y (Pixels)')
        ax.grid(False) # Grid usually looks messy over an image
        
        # Only manually invert Y if we didn't plot an image.
        # imshow sets Y=0 at the top; default plot sets Y=0 at the bottom.
        if not bg_image:
            ax.invert_yaxis()
            ax.grid(True)
        
        InteractivePlotter._setup_checkboxes(fig, plots, labels)
        
        logger.info(f"Opening interactive 2D plot: {title}")
        plt.show(block=block)

    @staticmethod
    def _setup_checkboxes(fig, plots, labels):
        """Shared logic for setting up visibility checkboxes."""
        ax_check = plt.axes([0.05, 0.4, 0.15, 0.2])  # [left, bottom, width, height]
        check = CheckButtons(ax_check, labels, [True] * len(labels))

        def toggle_visibility(label):
            index = labels.index(label)
            plots[index].set_visible(not plots[index].get_visible())
            plt.draw()

        check.on_clicked(toggle_visibility)
        # Keep reference
        fig._check_buttons_ref = check

# -----------------------------------------------------
# Controller: Main Application Logic
# -----------------------------------------------------
class HandTrackingClient:
    def __init__(self, config: AppConfig):
        self.config = config
        self.client = ClientAPI(config.base_url)
        # Ensure output directory exists
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, input_path: Path):
        """Orchestrates the processing pipeline."""
        if not input_path.exists():
            logger.error(f"Input path does not exist: {input_path}")
            return

        try:
            # 1. Upload & Process
            data, is_video = self._process_file(input_path)
            
            # 2. Validation
            if not self._validate_response(data):
                return

            logger.info("Response received successfully.")

            # 3. Save Results
            self._save_results(data, is_video)

            # 4. Visualization
            self._visualize_results(data, is_video)

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error: {e}")
        except json.JSONDecodeError as e:
            logger.error(f"API returned invalid JSON: {e}")
        except Exception as e:
            logger.exception(f"Unexpected error: {e}")

    def _process_file(self, path: Path) -> Tuple[Dict[str, Any], bool]:
        """Determines file type and uploads to correct endpoint."""
        ext = path.suffix.lower()
        if ext in [".jpg", ".jpeg", ".png"]:
            logger.info(f"Sending image {path} ...")
            return self.client.upload_image(str(path)), False
        elif ext in [".mp4", ".avi", ".mov", ".mkv"]:
            logger.info(f"Sending video {path} ...")
            return self.client.upload_video(str(path)), True
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def _validate_response(self, data: Optional[Dict[str, Any]]) -> bool:
        if not data:
            logger.error("API returned no data.")
            return False
        if data.get('error'):
            logger.error(f"API returned error: {data['error']}")
            return False
        return True

    def _save_results(self, data: Dict[str, Any], is_video: bool):
        prefix = "video_output" if is_video else "image_output"
        
        # Save JSON
        timestamp = results_handler.save_json_output(
            data, str(self.config.output_dir), prefix
        )

        # Save Artifacts via Result Handler
        if is_video:
            results_handler.save_video_response(
                data, str(self.config.output_dir), timestamp
            )
            self._log_video_stats(data)
        else:
            results_handler.save_image_response(
                data, str(self.config.output_dir), timestamp
            )
            count = len(data.get('hands', []))
            logger.info(f"Detected {count} hands.")

    def _log_video_stats(self, data: Dict[str, Any]):
        samples = data.get('samples', [])
        total_frames = len(samples)
        frames_with_hands = sum(1 for f in samples if f.get("hands"))
        logger.info(f"Video Stats - Total Frames: {total_frames}, Frames with Hands: {frames_with_hands}")

    def _visualize_results(self, data: Dict[str, Any], is_video: bool):
        """
        Visualizes the results in 4 windows:
        1. Rendered Image (2D)
        2. Standard 3D Vertices
        3. Planar Z=0 Vertices
        4. Pixel Space Vertices (XY Plane)
        """
        # 1. Image Visualization
        images, title = DataExtractor.extract_images(data, is_video)
        
        # Retain the primary image for the 4th plot background
        primary_bg_image = images[0] if images else None

        if images:
            logger.info("Displaying rendered 2D image(s)...")
            visualization.show_image_result(images, title, blocking=False)
        else:
            logger.info("No rendered 2D images available to display.")

        # 2. 3D Visualization (Standard)
        vertices = DataExtractor.extract_vertices(data, is_video, key='vertices_3d', apply_translation=False)
        if vertices:
            logger.info("Displaying 3D vertices (Interactive)...")
            InteractivePlotter.plot_3d_vertices(vertices, title='3D Hand Vertices', block=False)
        else:
            logger.info("No 3D vertex data available to display.")

        # 3. 3D Visualization (Planar Z=0)
        vertices_planar = DataExtractor.extract_vertices(data, is_video, key='vertices_planar_z0', apply_translation=False)
        if vertices_planar:
            logger.info("Displaying Planar Z=0 vertices (Interactive)...")
            # Set block=False here to allow the 4th window to open
            InteractivePlotter.plot_3d_vertices(vertices_planar, title='Planar Z=0 Hand Vertices', block=False)
        else:
            logger.info("No Planar Z=0 vertex data available to display.")
        
        # 4. 2D Visualization (Pixel Space)
        vertices_pixel = DataExtractor.extract_vertices(data, is_video, key='vertices_pixel', apply_translation=False)
        if vertices_pixel:
            logger.info("Displaying Pixel Space vertices (Interactive)...")
            # This is the final window, so we block execution here to keep windows open.
            # We inject the primary_bg_image here.
            InteractivePlotter.plot_2d_vertices(
                vertices_pixel, 
                title='Hand Vertices (Pixel Plane)', 
                bg_image=primary_bg_image,
                block=True
            )
        else:
            logger.info("No Pixel Space vertex data available to display.")
            # If this logic is hit, the script might exit immediately if all previous were block=False.
            if vertices or vertices_planar or images:
                plt.show(block=True)

# -----------------------------------------------------
# Entry Point
# -----------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Client for Hand Tracking API")
    parser.add_argument("--input", help="Path to input file.")
    parser.add_argument("--host", default="localhost", help="API Host")
    parser.add_argument("--port", default="8080", help="API Port")
    
    args = parser.parse_args()

    # Initialize Config
    config = AppConfig(
        host=args.host,
        port=args.port
    )
    
    logger.info(f"Targeting API at {config.base_url}")

    # Resolve Input Path
    input_path = InputManager.get_input_path(args.input, config.default_input_dir)
    
    if not input_path:
        logger.info("Process cancelled by user.")
        return

    # Execute Logic
    app = HandTrackingClient(config)
    app.run(input_path)

if __name__ == "__main__":
    main()