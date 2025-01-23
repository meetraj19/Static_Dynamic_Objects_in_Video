import cv2
import numpy as np


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


# Main function to perform decoupled representation
def decoupled_representation(video_path):
    # Step 1: Extract frames from the video
    frames = extract_frames(video_path)

    for i, frame in enumerate(frames):
        print(f"Processing frame {i + 1}/{len(frames)}")

        # Step 2: Segment the frame into dynamic and static parts
        dynamic_mask, static_mask = segment_dynamic_static(frame)

        # Step 3: Visualize the segmentation
        dynamic_output = cv2.bitwise_and(frame, frame, mask=dynamic_mask)
        static_output = cv2.bitwise_and(frame, frame, mask=static_mask)

        # Display the results
        cv2.imshow('Original Frame', frame)
        cv2.imshow('Dynamic Part', dynamic_output)
        cv2.imshow('Static Part', static_output)

        # Press 'q' to exit early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = '/home/thales1/VSLAM/Vslam_algorithm/pythonProject1/Visual_slam_optimized/DATA/istockphoto-1273783707-640_adpp_is.mp4'  # Replace with your actual video file path
    decoupled_representation(video_path)
