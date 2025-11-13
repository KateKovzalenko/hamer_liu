import os
import json
import base64
from io import BytesIO
from PIL import Image
from datetime import datetime

def _get_timestamp():
    """Returns a standardized timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def save_json_output(data, output_dir, prefix):
    """
    Saves the full JSON response to a file.
    
    Parameters:
    data (dict): The JSON data to save.
    output_dir (str): The directory to save in.
    prefix (str): The prefix for the filename (e.g., "image_output").
    
    Returns:
    str: The timestamp used for the filename.
    """
    timestamp = _get_timestamp()
    json_filename = os.path.join(output_dir, f"{prefix}_{timestamp}.json")
    with open(json_filename, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Full response saved to {json_filename}")
    return timestamp

def save_image_response(data, output_dir, timestamp):
    """
    Saves rendered images from an /upload/image response.
    This function *only* saves files and returns nothing.
    """
    images_to_save = []
    
    # Prefer single full-frame image if present
    full_b64 = data.get("full_frame_rendered_image_base64")
    if full_b64:
        img_data = base64.b64decode(full_b64)
        img = Image.open(BytesIO(img_data))
        image_filename = os.path.join(output_dir, f"full_frame_{timestamp}.png")
        img.save(image_filename)
        print(f"Full-frame rendered image saved as {image_filename}")
    else:
        # backward-compatible: per-hand rendered images inside hands[]
        if "hands" in data and data["hands"]:
            for hand in data["hands"]:
                if "rendered_image_base64" in hand:
                    img_data = base64.b64decode(hand["rendered_image_base64"])
                    images_to_save.append(Image.open(BytesIO(img_data)))

        if images_to_save:
            # Save a combined image
            widths, heights = zip(*(img.size for img in images_to_save))
            total_width = sum(widths)
            max_height = max(heights)
            combined = Image.new("RGB", (total_width, max_height))
            x_offset = 0
            for img in images_to_save:
                combined.paste(img, (x_offset, 0))
                x_offset += img.width
            
            image_filename = os.path.join(output_dir, f"all_hands_output_{timestamp}.png")
            combined.save(image_filename)
            print(f"Rendered per-hand image(s) saved as {image_filename}")
        else:
            print(f"No rendered image found to save. Hands detected: {len(data.get('hands', []))}")
            
    # This function intentionally returns nothing.

def save_video_response(data, output_dir, timestamp):
    """
    Saves rendered images from an /upload/video response.
    This function *only* saves files and returns nothing.
    """
    images_to_save = []

    # Prefer top-level full-frame image if present
    full_b64 = data.get("full_frame_rendered_image_base64")
    if full_b64:
        img_data = base64.b64decode(full_b64)
        img = Image.open(BytesIO(img_data))
        image_filename = os.path.join(output_dir, f"video_full_frame_{timestamp}.png")
        img.save(image_filename)
        print(f"Full-frame rendered image saved as {image_filename}")
    else:
        # fallback: check first sample for per-hand rendered images
        first_sample = data.get("samples", [None])[0]
        if first_sample and "hands" in first_sample:
            for hand in first_sample["hands"]:
                if "rendered_image_base64" in hand:
                    img_data = base64.b64decode(hand["rendered_image_base64"])
                    images_to_save.append(Image.open(BytesIO(img_data)))

        if images_to_save:
            # Save a combined image
            widths, heights = zip(*(img.size for img in images_to_save))
            total_width = sum(widths)
            max_height = max(heights)
            combined = Image.new("RGB", (total_width, max_height))
            x_offset = 0
            for img in images_to_save:
                combined.paste(img, (x_offset, 0))
                x_offset += img.width
                
            image_filename = os.path.join(output_dir, f"first_frame_{timestamp}.png")
            combined.save(image_filename)
            print(f"Rendered first frame saved as {image_filename}")
        else:
            print("No rendered images available to save. Samples returned:", len(data.get("samples", [])))
            
    # This function intentionally returns nothing.