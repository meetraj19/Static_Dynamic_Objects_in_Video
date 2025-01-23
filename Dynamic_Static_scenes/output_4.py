import numpy as np
import cv2
import os


# Function to load the fused volume from .npy file
def load_fused_volume(fused_volume_path):
    print(f"Loading fused volume from: " + fused_volume_path)
    fused_volume = np.load(fused_volume_path)
    return fused_volume


# Function to apply perspective projection on a 3D volume slice
def perspective_projection(fused_volume, f, original_frame_shape, output_folder, max_image_size=(2000, 2000), min_scale_factor=0.1):
    height, width, depth = fused_volume.shape

    # Initialize a cumulative change volume
    cumulative_volume = np.zeros(original_frame_shape, dtype=np.uint8)

    for z in range(depth):
        slice_image = fused_volume[:, :, z]
        z_distance = z + 1  # Avoid division by zero
        scale_factor = max(min(f / z_distance, 1), min_scale_factor)  # Scale factor between min_scale_factor and 1

        # Calculate the resized image dimensions
        resized_image_height = int(height * scale_factor)
        resized_image_width = int(width * scale_factor)

        if resized_image_height > max_image_size[0] or resized_image_width > max_image_size[1]:
            print(f"Skipping frame {z} due to excessive image size after resizing.")
            continue

        # Resizing the image slice according to the adjusted scale factor
        resized_image = cv2.resize(slice_image, (resized_image_width, resized_image_height))

        # Determine the placement of the resized image on the final projected image
        center_x = (original_frame_shape[1] - resized_image.shape[1]) // 2
        center_y = (original_frame_shape[0] - resized_image.shape[0]) // 2

        # Ensure the resized image fits within the projected image
        start_x = max(center_x, 0)
        start_y = max(center_y, 0)
        end_x = min(center_x + resized_image.shape[1], original_frame_shape[1])
        end_y = min(center_y + resized_image.shape[0], original_frame_shape[0])

        img_start_x = max(-center_x, 0)
        img_start_y = max(-center_y, 0)
        img_end_x = img_start_x + (end_x - start_x)
        img_end_y = img_start_y + (end_y - start_y)

        # Ensure the dimensions match before performing the max operation
        if img_end_x <= img_start_x or img_end_y <= img_start_y:
            continue  # Skip this slice if the resulting region is invalid

        # Update the cumulative volume directly without resizing
        cumulative_volume[start_y:end_y, start_x:end_x] = cv2.max(
            cumulative_volume[start_y:end_y, start_x:end_x],
            resized_image[img_start_y:img_end_y, img_start_x:img_end_x])

        # Save the projected image slice with cumulative changes
        output_path = os.path.join(output_folder, f'cumulative_projected_slice_{z}.png')
        cv2.imwrite(output_path, cumulative_volume)
        print(f"Saved cumulative projected image slice {z} to {output_path}")


# Main function for rendering and saving projected images with cumulative changes
def rendering_projection_process(save_folder, original_frame_shape, focal_length, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Step 1: Load the fused volume
    fused_volume_path = os.path.join(save_folder, 'fused_volume.npy')
    print("Loading fused volume...")
    fused_volume = load_fused_volume(fused_volume_path)

    # Step 2: Apply perspective projection and save images with cumulative changes
    print("Applying perspective projection with cumulative changes...")
    perspective_projection(fused_volume, focal_length, original_frame_shape, output_folder)
    print("Projection and saving with cumulative changes completed.")


if __name__ == "__main__":
    save_folder = '/home/thales1/VSLAM/Vslam_algorithm/pythonProject1/Visual_slam_optimized/Dynamic_Static_scenes/fused_volume'  # Specify the folder where the volumes are saved
    output_folder = '/home/thales1/VSLAM/Vslam_algorithm/pythonProject1/Visual_slam_optimized/Dynamic_Static_scenes/output/cumulative_images'  # Specify the folder where cumulative projected images will be saved
    original_frame_shape = (720, 1280)  # Replace with your original frame shape (height, width)
    focal_length = 300  # Set the focal length for perspective projection
    rendering_projection_process(save_folder, original_frame_shape, focal_length, output_folder)
