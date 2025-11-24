import numpy as np
from PIL import Image


def replace_background(plot_image_path, color_image_path, output_path, tolerance=5):
    """
    Replace white background in a plot image with a color from another image.

    Parameters:
    -----------
    plot_image_path : str
        Path to the plot image with white background
    color_image_path : str
        Path to the image containing the new background color
    output_path : str
        Path where the output image will be saved
    tolerance : int
        Tolerance for matching background pixels (default: 5)
    """
    # Load images
    plot_img = Image.open(plot_image_path).convert('RGB')
    color_img = Image.open(color_image_path).convert('RGB')

    # Convert to numpy arrays
    plot_array = np.array(plot_img, dtype=np.int16)  # Use int16 to avoid overflow
    color_array = np.array(color_img)

    # Get the background color from position (0,0) of the plot
    bg_color = plot_array[0, 0].astype(np.int16)

    # Get the new color from the color image (take the first pixel)
    new_color = color_array[0, 0]

    # Debug: print the background color
    print(f"Background color detected: {bg_color}")
    print(f"New color: {new_color}")

    # Create a mask for pixels that match the background color
    # Calculate absolute difference for each channel
    diff_r = np.abs(plot_array[:, :, 0] - bg_color[0])
    diff_g = np.abs(plot_array[:, :, 1] - bg_color[1])
    diff_b = np.abs(plot_array[:, :, 2] - bg_color[2])

    # Only pixels where ALL channels are within tolerance
    mask = (diff_r <= tolerance) & (diff_g <= tolerance) & (diff_b <= tolerance)

    # Debug: check some pixel values
    print(f"Number of background pixels found: {np.sum(mask)}")
    print(f"Total pixels: {mask.size}")

    # Convert back to uint8 for processing
    plot_array = plot_array.astype(np.uint8)

    # Replace background pixels with the new color
    result_array = plot_array.copy()
    result_array[mask] = new_color

    # Convert back to image and save
    result_img = Image.fromarray(result_array)
    result_img.save(output_path)

    return result_img


import numpy as np
from PIL import Image


def make_background_transparent(image_path, output_path, tolerance=5):
    """
    Replace white background with transparency.

    Parameters:
    -----------
    image_path : str
        Path to the image with white background
    output_path : str
        Path where the output image will be saved (must be PNG)
    tolerance : int
        Tolerance for matching background pixels (default: 5)

    Returns:
    --------
    PIL.Image
        The image with transparent background
    """
    # Load image and convert to RGBA
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img, dtype=np.int16)

    # Get the background color from position (0,0) - assuming it's white
    bg_color = img_array[0, 0].astype(np.int16)

    print(f"Background color detected: {bg_color}")

    # Create a mask for background pixels (white or near-white)
    diff_r = np.abs(img_array[:, :, 0] - bg_color[0])
    diff_g = np.abs(img_array[:, :, 1] - bg_color[1])
    diff_b = np.abs(img_array[:, :, 2] - bg_color[2])

    # Pixels that match background (within tolerance)
    bg_mask = (diff_r <= tolerance) & (diff_g <= tolerance) & (diff_b <= tolerance)

    print(f"Number of background pixels found: {np.sum(bg_mask)}")
    print(f"Total pixels: {bg_mask.size}")

    # Convert back to uint8 for processing
    img_array = img_array.astype(np.uint8)

    # Create RGBA array
    rgba_array = np.zeros((img_array.shape[0], img_array.shape[1], 4), dtype=np.uint8)
    rgba_array[:, :, :3] = img_array  # Copy RGB channels

    # Set alpha channel: 0 (transparent) for background, 255 (opaque) for content
    rgba_array[:, :, 3] = np.where(bg_mask, 0, 255)

    # Convert to image and save
    result_img = Image.fromarray(rgba_array, mode='RGBA')
    result_img.save(output_path, format='PNG')

    print(f"Transparent background image saved to {output_path}")
    return result_img


def crop_white_background(image_path, output_path=None, tolerance=5):
    """
    Crop white background from an image, keeping only the content area.

    Parameters:
    -----------
    image_path : str
        Path to the image with white background
    output_path : str, optional
        Path where the cropped image will be saved. If None, returns the image without saving.
    tolerance : int
        Tolerance for matching background pixels (default: 5)

    Returns:
    --------
    PIL.Image
        The cropped image
    """
    # Load image
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img, dtype=np.int16)

    # Get the background color from position (0,0)
    bg_color = img_array[0, 0].astype(np.int16)

    print(f"Background color detected: {bg_color}")

    # Create a mask for non-background pixels
    diff_r = np.abs(img_array[:, :, 0] - bg_color[0])
    diff_g = np.abs(img_array[:, :, 1] - bg_color[1])
    diff_b = np.abs(img_array[:, :, 2] - bg_color[2])

    # Pixels that DON'T match background (content pixels)
    content_mask = (diff_r > tolerance) | (diff_g > tolerance) | (diff_b > tolerance)

    # Find the bounding box of content pixels
    rows = np.any(content_mask, axis=1)
    cols = np.any(content_mask, axis=0)

    # Get the indices where content exists
    row_indices = np.where(rows)[0]
    col_indices = np.where(cols)[0]

    if len(row_indices) == 0 or len(col_indices) == 0:
        print("Warning: No content found in image!")
        return img

    # Get bounding box coordinates
    top = row_indices[0]
    bottom = row_indices[-1] + 1 - 220
    left = col_indices[0]
    right = col_indices[-1] + 1

    print(f"Cropping to: top={top}, bottom={bottom}, left={left}, right={right}")
    print(f"Original size: {img.size}, New size: {(right - left, bottom - top)}")

    # Crop the image
    cropped_img = img.crop((left, top, right, bottom))

    # Save if output path is provided
    if output_path:
        cropped_img.save(output_path)
        print(f"Cropped image saved to {output_path}")

    return cropped_img


# Example usage
if __name__ == "__main__":
    # Make background transparent
    make_background_transparent(
        image_path='unsupervised_directions.png',
        output_path='unsupervised_directions_transparent.png',
        tolerance=5
    )
# # Example usage
# if __name__ == "__main__":
#     # Replace 'plot.png', 'color.png', and 'output.png' with your file paths
#     replace_background(
#         plot_image_path='training.jpg',
#         color_image_path='background_img.png',
#         output_path='training_gray.png',
#         tolerance=2
#     )
#     print("Background replacement complete!")