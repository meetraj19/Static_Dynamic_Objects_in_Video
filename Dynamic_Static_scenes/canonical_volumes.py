import cv2
import numpy as np
import os


# Function to load masks from saved files
def load_masks(dynamic_mask_folder, static_mask_folder, frame_index):
    dynamic_mask_path = os.path.join(dynamic_mask_folder, f'dynamic_mask_{frame_index}.png')
    static_mask_path = os.path.join(static_mask_folder, f'static_mask_{frame_index}.png')

    dynamic_mask = cv2.imread(dynamic_mask_path, cv2.IMREAD_GRAYSCALE)
    static_mask = cv2.imread(static_mask_path, cv2.IMREAD_GRAYSCALE)

    return dynamic_mask, static_mask


# Function to initialize a 3D volume
def initialize_volume(width, height, depth):
    return np.zeros((height, width, depth), dtype=np.uint8)


# Function to update the canonical volumes
def update_volume(volume, transformed_part, frame_index):
    # Add the transformed 2D part as a slice in the 3D volume
    volume[:, :, frame_index] = transformed_part
    return volume


# Main function to create canonical volumes
def canonical_volumes_process(video_path, dynamic_mask_folder, static_mask_folder, save_folder):
    # Ensure the save folder exists
    os.makedirs(save_folder, exist_ok=True)

    # Step 1: Extract frames from the video
    cap = cv2.VideoCapture(video_path)
    frame_index = 0

    # Get video dimensions
    ret, frame = cap.read()
    height, width, _ = frame.shape
    cap.release()

    # Initialize canonical volumes for dynamic and static parts
    depth = 100  # keep 100 frames in the volume
    dynamic_volume = initialize_volume(width, height, depth)
    static_volume = initialize_volume(width, height, depth)

    cap = cv2.VideoCapture(video_path)

    while cap.isOpened() and frame_index < depth:
        ret, frame = cap.read()
        if not ret:
            break

        print(f"Processing frame {frame_index + 1}")

        # Step 2: Load the saved masks
        dynamic_mask, static_mask = load_masks(dynamic_mask_folder, static_mask_folder, frame_index)

        # Apply the dynamic and static masks to extract the parts
        dynamic_part = cv2.bitwise_and(frame, frame, mask=dynamic_mask)
        static_part = cv2.bitwise_and(frame, frame, mask=static_mask)

        # Convert the dynamic and static parts to grayscale for volume update
        dynamic_gray = cv2.cvtColor(dynamic_part, cv2.COLOR_BGR2GRAY)
        static_gray = cv2.cvtColor(static_part, cv2.COLOR_BGR2GRAY)

        # Step 3: Update canonical volumes with the transformed parts
        dynamic_volume = update_volume(dynamic_volume, dynamic_gray, frame_index)
        static_volume = update_volume(static_volume, static_gray, frame_index)

        frame_index += 1

    cap.release()

    # Save the volumes
    dynamic_volume_path = os.path.join(save_folder, 'dynamic_volume.npy')
    static_volume_path = os.path.join(save_folder, 'static_volume.npy')

    print("Saving dynamic volume...")
    np.save(dynamic_volume_path, dynamic_volume)

    print("Saving static volume...")
    np.save(static_volume_path, static_volume)

    print(f"Dynamic and static volumes saved to {save_folder}")
    print("Process completed successfully. Exiting now.")


if __name__ == "__main__":
    video_path = '/home/thales1/VSLAM/Vslam_algorithm/pythonProject1/Visual_slam_optimized/DATA/istockphoto-1250036378-640_adpp_is.mp4'  # Replace with your actual video file path
    dynamic_mask_folder = '/home/thales1/VSLAM/Vslam_algorithm/pythonProject1/Visual_slam_optimized/Dynamic_Static_scenes/dynamic_masks'  # Specify the folder where dynamic masks are saved
    static_mask_folder = '/home/thales1/VSLAM/Vslam_algorithm/pythonProject1/Visual_slam_optimized/Dynamic_Static_scenes/static_masks'  # Specify the folder where static masks are saved
    save_folder = '/home/thales1/VSLAM/Vslam_algorithm/pythonProject1/Visual_slam_optimized/Dynamic_Static_scenes/volumes'  # Specify the folder where volumes will be saved
    canonical_volumes_process(video_path, dynamic_mask_folder, static_mask_folder, save_folder)
