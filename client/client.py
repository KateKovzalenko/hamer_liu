import argparse
import requests
import base64
import matplotlib.pyplot as plt
from PIL import Image
import io
from io import BytesIO
import os
import json
from datetime import datetime

# -----------------------------------------------------
# Configuration
# -----------------------------------------------------
DEFAULT_HOST = "localhost"
DEFAULT_PORT = "8080"

def send_image(image_path, api_url):
    """Send an image file to the /upload/image endpoint and show/save the rendered result."""

    # --- Ensure output folder exists ---
    output_dir = "demo_out"
    os.makedirs(output_dir, exist_ok=True)

    # --- Upload image ---
    with open(image_path, 'rb') as f:
        files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
        print(f"Sending image to {api_url} ...")
        response = requests.post(api_url, files=files, timeout=(10, 600)) # 10s connect timeout, 600s read timeout (10 minutes)
    response.raise_for_status()

    result = response.json()

    # --- Save full JSON response with unique timestamp ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_filename = os.path.join(output_dir, f"full_output_{timestamp}.json")
    with open(json_filename, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Full response saved to {json_filename}")

    # --- Handle hands rendered images ---
    if "hands" in result and result["hands"]:
        images = []
        for hand in result["hands"]:
            if "rendered_image_base64" in hand:
                img_data = base64.b64decode(hand["rendered_image_base64"])
                images.append(Image.open(BytesIO(img_data)))

        if images:
            # Combine images horizontally (side by side)
            widths, heights = zip(*(img.size for img in images))
            total_width = sum(widths)
            max_height = max(heights)
            combined = Image.new("RGB", (total_width, max_height))
            x_offset = 0
            for img in images:
                combined.paste(img, (x_offset, 0))
                x_offset += img.width

            image_filename = os.path.join(output_dir, f"all_hands_output_{timestamp}.png")
            combined.show(title="HAMER Output")
            combined.save(image_filename)
            print(f"Rendered image(s) saved as {image_filename}")
        else:
            print("No rendered images found, but vertices exist.")
    else:
        print("No rendered image found in response. Got vertices only.")

    return result


def send_video(video_path, api_url):
    """Send a video file to the /upload/video endpoint."""
    with open(video_path, 'rb') as f:
        files = {'file': (os.path.basename(video_path), f, 'video/mp4')}
        print(f"Sending video to {api_url} ...")
        response = requests.post(api_url, files=files, timeout=60)
    response.raise_for_status()
    return response.json(), None

def plot_vertices(image_bytes, vertices):
    """Plot hand tracking landmarks on the input image."""
    img = Image.open(io.BytesIO(image_bytes))
    img_width, img_height = img.size

    _, ax = plt.subplots(1)
    ax.imshow(img)

    for v in vertices:
        x = v['x'] * img_width
        y = v['y'] * img_height
        ax.plot(x, y, 'ro', markersize=3)

    plt.axis('off')
    plt.title("Hand Tracking Results")
    plt.show()


# -----------------------------------------------------
# Main CLI
# -----------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Client for Hand Tracking API (auto mode)")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input file or directory."
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

    input_path = args.input
    base_url = f"http://{args.host}:{args.port}"

    try:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input path does not exist: {input_path}")

        # --- Decide mode automatically ---
        if os.path.isfile(input_path):
            ext = os.path.splitext(input_path)[1].lower()
            if ext in [".jpg", ".jpeg", ".png"]:
                mode = "image"
                api_url = f"{base_url}/upload/image"
                data = send_image(input_path, api_url)
                img_bytes = None
                #data, img_bytes = send_image(input_path, api_url)
            elif ext in [".mp4", ".avi", ".mov", ".mkv"]:
                mode = "video"
                api_url = f"{base_url}/upload/video"
                data = send_video(input_path, api_url)
            else:
                raise ValueError(f"Unsupported file type: {ext}")

        else:
            raise ValueError(f"Invalid input: {input_path}")

        # --- Handle response ---
        print(" Response received:")
        print(json.dumps(data, indent=2))

        hands = data.get("hands", [])
        # TODO: Output received base64 image.       

    except requests.exceptions.RequestException as e:
        print(f" Request failed: {e}")
    except json.JSONDecodeError as e:
        print(f" JSON decode error: {e}")
    except FileNotFoundError as e:
        print(f" Input error: {e}")
    except ValueError as e:
        print(f" File type error: {e}")
    except OSError as e:
        print(f" OS error: {e}")


if __name__ == "__main__":
    main()
