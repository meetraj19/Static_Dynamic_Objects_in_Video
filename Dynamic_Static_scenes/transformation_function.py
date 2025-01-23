import cv2
import numpy as np
import os


# Function to extract frames from the video
def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


# Placeholder function to segment dynamic and static parts
def segment_dynamic_static(frame):
    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Placeholder segmentation: Static parts as darker regions, dynamic as brighter regions
    _, dynamic_mask = cv2.threshold(gray_frame, 128, 255, cv2.THRESH_BINARY)
    static_mask = cv2.bitwise_not(dynamic_mask)

    return dynamic_mask, static_mask


# Function to apply affine transformation
def apply_transformation(image, matrix):
    # Apply the transformation matrix to the image
    transformed_image = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))
    return transformed_image


# Main function to perform decoupled representation and transformation
def decoupled_transformation(video_path, dynamic_folder, static_folder):
    # Ensure the mask folders exist
    os.makedirs(dynamic_folder, exist_ok=True)
    os.makedirs(static_folder, exist_ok=True)

    # Step 1: Extract frames from the video
    frames = extract_frames(video_path)

    # Define transformation matrices (e.g., rotation, translation)
    angle = 20
    rotation_matrix = cv2.getRotationMatrix2D((frames[0].shape[1] // 2, frames[0].shape[0] // 2), angle, 1)

    translation_matrix = np.float32([[1, 0, 50], [0, 1, 30]])  # Translate by (50, 30) pixels

    for i, frame in enumerate(frames):
        print(f"Processing frame {i + 1}/{len(frames)}")

        # Step 2: Segment the frame into dynamic and static parts
        dynamic_mask, static_mask = segment_dynamic_static(frame)

        # Save the masks to folders
        dynamic_mask_path = os.path.join(dynamic_folder, f'dynamic_mask_{i}.png')
        static_mask_path = os.path.join(static_folder, f'static_mask_{i}.png')
        cv2.imwrite(dynamic_mask_path, dynamic_mask)
        cv2.imwrite(static_mask_path, static_mask)

        # Apply the dynamic and static masks to extract the parts
        dynamic_part = cv2.bitwise_and(frame, frame, mask=dynamic_mask)
        static_part = cv2.bitwise_and(frame, frame, mask=static_mask)

        # Step 3: Apply transformations to the dynamic and static parts
        transformed_dynamic = apply_transformation(dynamic_part, rotation_matrix)
        transformed_static = apply_transformation(static_part, translation_matrix)

        # Display the original, dynamic, and static transformed frames
        cv2.imshow('Original Frame', frame)
        cv2.imshow('Transformed Dynamic Part', transformed_dynamic)
        cv2.imshow('Transformed Static Part', transformed_static)

        # Press 'q' to exit early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = '/home/thales1/VSLAM/Vslam_algorithm/pythonProject1/Visual_slam_optimized/DATA/istockphoto-1250036378-640_adpp_is.mp4'  # Replace with your actual video file path
    dynamic_folder = '/home/thales1/VSLAM/Vslam_algorithm/pythonProject1/Visual_slam_optimized/Dynamic_Static_scenes/masks/dynamic_masks'  # Specify the folder where dynamic masks will be saved
    static_folder = '/home/thales1/VSLAM/Vslam_algorithm/pythonProject1/Visual_slam_optimized/Dynamic_Static_scenes/masks/static_masks'  # Specify the folder where static masks will be saved
    decoupled_transformation(video_path, dynamic_folder, static_folder)
