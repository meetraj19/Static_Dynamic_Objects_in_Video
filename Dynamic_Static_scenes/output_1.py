import cv2
import os


# Function to create a video from projected images
def create_video_from_images(image_folder, output_video_path, frame_shape, fps=10):
    # Get the list of all image files in the directory
    images = sorted([img for img in os.listdir(image_folder) if img.endswith(".png")])

    if not images:
        print("No images found in the specified folder.")
        return

    # Define the video codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, frame_shape)

    for image_name in images:
        image_path = os.path.join(image_folder, image_name)
        frame = cv2.imread(image_path)
        frame_resized = cv2.resize(frame, frame_shape)

        video_writer.write(frame_resized)
        print(f"Added {image_name} to video.")

    video_writer.release()
    print(f"Video saved as {output_video_path}")


if __name__ == "__main__":
    image_folder = '/home/thales1/VSLAM/Vslam_algorithm/pythonProject1/Visual_slam_optimized/Dynamic_Static_scenes/projected_image'  # Folder containing the projected images
    output_video_path = '/home/thales1/VSLAM/Vslam_algorithm/pythonProject1/Visual_slam_optimized/Dynamic_Static_scenes/output/projected_video_1.avi'  # Output video file path
    frame_shape = (1280, 720)  # Frame shape (width, height)
    fps = 100 # Frames per second for the video

    create_video_from_images(image_folder, output_video_path, frame_shape, fps)
