import numpy as np
import cv2
import os


# Function to load the fused volume from .npy file
def load_fused_volume(fused_volume_path):
    print(f"Loading fused volume from: {fused_volume_path}")
    fused_volume = np.load(fused_volume_path)
    return fused_volume


# Function to apply perspective projection on a 3D volume slice
def perspective_projection(fused_volume, f, original_frame_shape, output_folder, max_image_size=(2000, 2000)):
    height, width, depth = fused_volume.shape

    for z in range(depth):
        slice_image = fused_volume[:, :, z]
        z_distance = z + 1  # Avoid division by zero
        scale_factor = f / z_distance

        # Dynamically adjust the scale factor to ensure the resized image does not exceed max_image_size
        resized_image_height = int(height * scale_factor)
        resized_image_width = int(width * scale_factor)

        if resized_image_height > max_image_size[0] or resized_image_width > max_image_size[1]:
            scale_factor = min(max_image_size[0] / height, max_image_size[1] / width)
            print(f"Adjusted scale factor for frame {z} to {scale_factor:.2f} to fit within max image size.")

        # Resizing the image slice according to the adjusted scale factor
        resized_image = cv2.resize(slice_image, (0, 0), fx=scale_factor, fy=scale_factor)

        # Create a blank image to hold the projected result
        projected_image = np.zeros(original_frame_shape, dtype=np.uint8)

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

        # Overlay the resized image onto the projected image
        projected_image[start_y:end_y, start_x:end_x] = cv2.max(
            projected_image[start_y:end_y, start_x:end_x],
            resized_image[img_start_y:img_end_y, img_start_x:img_end_x])

        # Save the projected image slice
        output_path = os.path.join(output_folder, f'projected_slice_{z}.png')
        cv2.imwrite(output_path, projected_image)
        print(f"Saved projected image slice {z} to {output_path}")


# Main function for rendering and saving projected images
def rendering_projection_process(save_folder, original_frame_shape, focal_length, output_folder):

    os.makedirs(output_folder, exist_ok=True)

    # Step 1: Load the fused volume
    fused_volume_path = os.path.join(save_folder, 'fused_volume.npy')
    print("Loading fused volume...")
    fused_volume = load_fused_volume(fused_volume_path)

    # Step 2: Apply perspective projection and save images
    print("Applying perspective projection...")
    perspective_projection(fused_volume, focal_length, original_frame_shape, output_folder)
    print("Projection and saving completed.")


if __name__ == "__main__":
    save_folder = '/home/thales1/VSLAM/Vslam_algorithm/pythonProject1/Visual_slam_optimized/Dynamic_Static_scenes/fused_volume'  # Specify the folder where the volumes are saved
    output_folder = '/home/thales1/VSLAM/Vslam_algorithm/pythonProject1/Visual_slam_optimized/Dynamic_Static_scenes/projected_image'  # Specify the folder where projected images will be saved
    original_frame_shape = (720, 1280)  # Replace with your original frame shape (height, width)
    focal_length = 500  # Set the focal length for perspective projection
    rendering_projection_process(save_folder, original_frame_shape, focal_length, output_folder)
