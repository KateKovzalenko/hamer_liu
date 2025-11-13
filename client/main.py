import argparse
import os
import json
import requests.exceptions
import tkinter as tk
from tkinter import filedialog
import base64
from io import BytesIO
from PIL import Image

from client_api import ClientAPI
import results_handler
import visualization

# -----------------------------------------------------
# Configuration
# -----------------------------------------------------
DEFAULT_HOST = "localhost"
DEFAULT_PORT = "8080"
DEFAULT_OUTPUT_DIR = "demo_out"

# -----------------------------------------------------
# User Input Handling
# -----------------------------------------------------

def get_input_path(cli_path):
    """
    Gets the input file path, either from the --input argument
    or by opening a GUI file dialog if --input is omitted.
    """
    if cli_path:
        return cli_path

    print("No --input provided, opening GUI file dialog...")
    root = tk.Tk()
    root.withdraw()  # Hide the main Tk window

    initial_dir = os.path.abspath("example_data")
    if not os.path.isdir(initial_dir):
        print(f"Warning: Initial directory '{initial_dir}' not found. Defaulting to current directory.")
        initial_dir = os.path.abspath(".")

    filetypes = [
        ("Media Files", ".jpg .jpeg .png .mp4 .avi .mov .mkv"),
        ("Image Files", ".jpg .jpeg .png"),
        ("Video Files", ".mp4 .avi .mov .mkv"),
        ("All Files", "*.*")
    ]

    input_path = filedialog.askopenfilename(
        initialdir=initial_dir,
        title="Select Image or Video File",
        filetypes=filetypes
    )

    if not input_path:
        print("No file selected. Exiting.")
        return None
    
    return input_path

def print_video_stats(data):
    """Prints statistics from a video response to the console."""
    total_frames = len(data.get('samples', []))
    print(f"Total frames processed: {total_frames}")
    frames_with_vertices = sum(1 for f in data.get('samples', []) if "hands" in f and f["hands"])
    print(f"Frames containing hand vertices: {frames_with_vertices}")

def is_valid_response(data):
    """
    Checks if the response data from the client is valid for processing.
    A valid response is not None and does not contain a top-level 'error' key.
    """
    if not data:
        print("API returned no data.")
        return False
    if data.get('error'):
        print(f"API returned an error: {data['error']}")
        return False
    return True

# -----------------------------------------------------
# Tier 2: Data Extraction for Plotting
# -----------------------------------------------------

def _decode_base64_to_pil(b64_string):
    """Helper to safely decode a base64 string into a PIL Image."""
    try:
        img_data = base64.b64decode(b64_string)
        return Image.open(BytesIO(img_data))
    except Exception as e:
        print(f" Warning: Could not decode base64 string: {e}")
        return None

def extract_rendered_images(data, is_video):
    """
    Extracts PIL images from the API response for 2D visualization.
    This logic intentionally mirrors 'results_handler' to decouple plotting.
    
    Returns:
    (list, str): A tuple containing a list of PIL.Image objects 
                 to display, and a title for the display window.
    """
    images_to_display = []
    display_title = "Render"

    # Check for full-frame image first (applies to both image and video)
    full_b64 = data.get("full_frame_rendered_image_base64")
    if full_b64:
        img = _decode_base64_to_pil(full_b64)
        if img:
            images_to_display = [img]
            display_title = "HAMER Full Frame"
        return images_to_display, display_title

    # If no full-frame, check for per-hand/per-frame images
    if not is_video:
        # --- Image Response Logic ---
        if "hands" in data and data["hands"]:
            for hand in data["hands"]:
                if "rendered_image_base64" in hand:
                    img = _decode_base64_to_pil(hand["rendered_image_base64"])
                    if img:
                        images_to_display.append(img)
            if images_to_display:
                display_title = "HAMER Hands"
    else:
        # --- Video Response Logic ---
        # fallback: check first sample for per-hand rendered images
        first_sample = data.get("samples", [None])[0]
        if first_sample and "hands" in first_sample:
            for hand in first_sample["hands"]:
                if "rendered_image_base64" in hand:
                    img = _decode_base64_to_pil(hand["rendered_image_base64"])
                    if img:
                        images_to_display.append(img)
            if images_to_display:
                display_title = "First frame render"

    return images_to_display, display_title

def extract_3d_vertices(data, is_video, *, apply_camera_translation: bool = False):
    """
    Extracts 3D vertex data from the API response for 3D plotting.
    Applies camera_translation to vertices if available.
    
    Returns:
    list: A list of point clouds (each cloud is a list of [x,y,z] points).
          Returns an empty list if no vertices are found.
    """
    point_clouds = []
    
    if not is_video:
        # --- Image Response Logic ---
        hands = data.get('hands', [])
        if hands:
            print(f"Found {len(hands)} hands in image response.")
            for i, hand in enumerate(hands):
                vertices = hand.get('vertices')
                translation = hand.get('camera_translation')
                
                if apply_camera_translation and vertices and translation:
                    # Apply translation if available and valid
                    if isinstance(translation, (list, tuple)) and len(translation) == 3:
                        try:
                            tx, ty, tz = translation
                            tz = tz/10
                            translated_vertices = [
                                [v[0] + tx, v[1] + ty, v[2] + tz] for v in vertices
                            ]
                            print(f"  Hand {i}: Found and translated {len(translated_vertices)} vertices.")
                            point_clouds.append(translated_vertices)
                        except (TypeError, IndexError) as e:
                            print(f"  Hand {i}: Error applying translation: {e}. Using original vertices.")
                            point_clouds.append(vertices) # Fallback
                    else:
                        print(f"  Hand {i}: Invalid 'camera_translation'. Using original vertices.")
                        point_clouds.append(vertices) # Fallback
                elif vertices:
                    # Use original vertices if no translation is found
                    print(f"  Hand {i}: Found {len(vertices)} vertices (no translation data).")
                    point_clouds.append(vertices)
                else:
                    print(f"  Hand {i}: No 'vertices' key found.")
    else:
        # --- Video Response Logic ---
        # Plot vertices from the *first* frame that has them
        first_sample_with_hands = None
        for sample in data.get('samples', []):
            if sample and sample.get('hands'):
                first_sample_with_hands = sample
                break # Found the first valid sample
        
        if first_sample_with_hands:
            hands = first_sample_with_hands.get('hands', [])
            print(f"Found {len(hands)} hands in first video frame with data.")
            for i, hand in enumerate(hands):
                vertices = hand.get('vertices')
                translation = hand.get('camera_translation')
                
                if apply_camera_translation and vertices and translation:
                    # Apply translation if available and valid
                    if isinstance(translation, (list, tuple)) and len(translation) == 3:
                        try:
                            tx, ty, tz = translation
                            translated_vertices = [
                                [v[0] + tx, v[1] + ty, v[2] + tz] for v in vertices
                            ]
                            print(f"  Hand {i}: Found and translated {len(translated_vertices)} vertices.")
                            point_clouds.append(translated_vertices)
                        except (TypeError, IndexError) as e:
                            print(f"  Hand {i}: Error applying translation: {e}. Using original vertices.")
                            point_clouds.append(vertices) # Fallback
                    else:
                        print(f"  Hand {i}: Invalid 'camera_translation'. Using original vertices.")
                        point_clouds.append(vertices) # Fallback
                elif vertices:
                    # Use original vertices if no translation is found
                    print(f"  Hand {i}: Found {len(vertices)} vertices (no translation data).")
                    point_clouds.append(vertices)
                else:
                    print(f"  Hand {i}: No 'vertices' key found.")
        else:
            print("No frames with hand data found in video response.")
            
    return point_clouds

# -----------------------------------------------------
# Main CLI
# -----------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Client for Hand Tracking API")
    parser.add_argument(
        "--input",
        required=False,
        default=None,
        help="Path to input file. If omitted, a GUI file selector will open."
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help="HOST of the running Flask API."
    )
    parser.add_argument(
        "--port",
        default=DEFAULT_PORT,
        help="PORT of the running Flask API."
    )

    args = parser.parse_args()

    print(f"Using API at {args.host}:{args.port}")

    input_path = get_input_path(args.input)
    if not input_path:
        return  # User cancelled GUI dialog

    base_url = f"http://{args.host}:{args.port}"
    client = ClientAPI(base_url)
    
    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

    try:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input path does not exist: {input_path}")

        ext = os.path.splitext(input_path)[1].lower()
        data = None
        is_video = False

        if ext in [".jpg", ".jpeg", ".png"]:
            print(f"Sending image {input_path} ...")
            data = client.upload_image(input_path)
            is_video = False

        elif ext in [".mp4", ".avi", ".mov", ".mkv"]:
            print(f"Sending video {input_path} ...")
            data = client.upload_video(input_path)
            is_video = True

        else:
            raise ValueError(f"Unsupported file type: {ext}")

        # --- Validation Step ---
        if not is_valid_response(data):
            print("Processing failed or API returned invalid data.")
            return

        print(" Response received.")

        # --- Tier 1: Results Saving ---
        # This tier now returns NO data for plotting
        timestamp = results_handler.save_json_output(
            data, DEFAULT_OUTPUT_DIR, "video_output" if is_video else "image_output"
        )
        
        if is_video:
            results_handler.save_video_response(
                data, DEFAULT_OUTPUT_DIR, timestamp
            )
            print_video_stats(data)
        else:
            results_handler.save_image_response(
                data, DEFAULT_OUTPUT_DIR, timestamp
            )
            print(f"  Detected {len(data.get('hands', []))} hands.")

        # --- Tier 2: Data Extraction ---
        # This tier extracts data from the raw 'data' dict for plotting
        images_to_display, display_title = extract_rendered_images(data, is_video)
        vertex_data = extract_3d_vertices(data, is_video, apply_camera_translation=True)

        # --- Tier 3: Visualization ---
        # This tier plots data from Tier 2
        if images_to_display:
            print("Displaying rendered 2D image(s)...")
            visualization.show_image_result(images_to_display, display_title, blocking=False)
        else:
            print("No rendered 2D images available to display.")

        if vertex_data:
            print("Displaying 3D vertices...")
            visualization.plot_3d_vertices(vertex_data, title='3D Hand Vertices', block=True)
        else:
            print("No 3D vertex data available to display.")


    except requests.exceptions.HTTPError as e:
        print(f" Request failed: {e}")
        if e.response is not None:
            print(f" Response text: {e.response.text}")
    except requests.exceptions.RequestException as e:
        print(f" Request connection failed: {e}")
    except json.JSONDecodeError as e:
        print(f" JSON decode error: {e}")
    except FileNotFoundError as e:
        print(f" Input error: {e}")
    except ValueError as e:
        print(f" File type error: {e}")
    except OSError as e:
        print(f" OS error: {e}")
    except Exception as e:
        print(f" An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()