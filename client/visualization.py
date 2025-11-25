import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from PIL import Image

# -----------------------------------------------------
# PIL Image Display
# -----------------------------------------------------

def _combine_images(images):
    """Stitches a list of PIL Images together horizontally."""
    if not images:
        return None
    
    widths, heights = zip(*(img.size for img in images))
    total_width = sum(widths)
    max_height = max(heights)
    
    combined = Image.new("RGB", (total_width, max_height))
    
    x_offset = 0
    for img in images:
        combined.paste(img, (x_offset, 0))
        x_offset += img.width
    return combined

def show_image_result(images, title, *, blocking: bool = True):
    """
    Displays one or more PIL images in a window using Matplotlib.
    If multiple images are provided, they are stitched together.
    
    Parameters:
    images (list): A list of PIL.Image objects.
    title (str): The title for the display window.
    blocking (bool): Whether the plot window should block execution.
    """
    if not images:
        print("No images to display.")
        return
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    if len(images) == 1:
        img_data = np.array(images[0])
    else:
        combined = _combine_images(images)
        if combined:
            img_data = np.array(combined)
        else:
            print("Failed to combine images for display.")
            plt.close(fig) # Close the empty figure
            return
            
    ax.imshow(img_data)
    ax.set_title(title)
    ax.axis('off') # Hide axes for cleaner image display
    plt.show(block=blocking) # Use the blocking parameter

# -----------------------------------------------------
# Matplotlib 3D Display
# -----------------------------------------------------

def plot_3d_vertices(data, title='3D Vertex Plot', *, block: bool = True):
    """
    Plots one or more 3D point clouds.

    Parameters:
    data: Can be one of two formats:
          1. A single point cloud: A list or Nx3 numpy array of [X, Y, Z] points.
             Example: [[1,2,3], [4,5,6], ...]
          2. A list of point clouds: A list of (list or Nx3 numpy array).
             Example: [ [[1,2,3], ...], [[7,8,9], ...] ]
    title (str): The title for the plot.
    """
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    point_clouds = []

    try:
        # Attempt to convert to a standard numpy array first
        arr = np.array(data)
        
        if arr.ndim == 2 and arr.shape[1] == 3:
            point_clouds = [arr]
        elif arr.ndim == 3:
             point_clouds = [arr[i] for i in range(arr.shape[0])]
        elif arr.ndim == 1 and arr.dtype == object:
            point_clouds = [np.array(pc) for pc in data]
            if not all(pc.ndim == 2 and pc.shape[1] == 3 for pc in point_clouds if pc.size > 0):
                 raise ValueError("One or more sub-clouds is not in Nx3 format.")
        else:
            raise ValueError("Input data format is not recognized.")

    except (ValueError, TypeError, IndexError) as e:
        print(f"Error processing input data: {e}")
        print("Please ensure data is either an Nx3 list/array or a list of Nx3 lists/arrays.")
        return

    if not point_clouds or all(pc.size == 0 for pc in point_clouds):
        print("No valid data to plot.")
        return

    # --- Plotting ---
    all_points = np.concatenate([pc for pc in point_clouds if pc.size > 0], axis=0)
    
    for i, cloud in enumerate(point_clouds):
        if cloud.size == 0:
            continue
            
        color = colors[i % len(colors)]
        # Swap Y and Z: (X, Z, Y) instead of (X, Y, Z)
        ax.scatter(cloud[:, 0], cloud[:, 2], cloud[:, 1], c=color, marker='o', label=f'Cloud {i}')

    # --- Set Labels and Title ---
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Z Coordinate')
    ax.set_zlabel('Y Coordinate')
    ax.set_title(title)
    if len(point_clouds) > 1:
        ax.legend()

    # --- Set equal aspect ratio ---
    x_coords_all = all_points[:, 0]
    # After swap: Y becomes Z, Z becomes Y
    z_coords_all = all_points[:, 1]  # original Y
    y_coords_all = all_points[:, 2]  # original Z

    max_range = np.array([x_coords_all.max() - x_coords_all.min(), 
                          y_coords_all.max() - y_coords_all.min(), 
                          z_coords_all.max() - z_coords_all.min()]).max() / 2.0

    if max_range == 0: # Handle case of a single point
        max_range = 1.0

    mid_x = (x_coords_all.max() + x_coords_all.min()) * 0.5
    mid_y = (y_coords_all.max() + y_coords_all.min()) * 0.5
    mid_z = (z_coords_all.max() + z_coords_all.min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show(block=block)