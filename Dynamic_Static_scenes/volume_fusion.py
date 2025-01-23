import numpy as np
import os


# Function to load volumes from saved .npy files
def load_volumes(dynamic_volume_path, static_volume_path):
    dynamic_volume = np.load(dynamic_volume_path)
    static_volume = np.load(static_volume_path)
    return dynamic_volume, static_volume


# Function to fuse the dynamic and static volumes
def fuse_volumes(dynamic_volume, static_volume):
    # Simple fusion by taking the maximum value at each voxel
    fused_volume = np.maximum(dynamic_volume, static_volume)
    return fused_volume


# Main function to load, fuse, and save volumes
def volume_fusion_process(save_folder):
    # Step 1: Load the canonical volumes
    dynamic_volume_path = os.path.join(save_folder, 'dynamic_volume.npy')
    static_volume_path = os.path.join(save_folder, 'static_volume.npy')

    print("Loading volumes...")
    dynamic_volume, static_volume = load_volumes(dynamic_volume_path, static_volume_path)

    # Step 2: Fuse the volumes
    print("Fusing volumes...")
    fused_volume = fuse_volumes(dynamic_volume, static_volume)

    # Step 3: Save the fused volume
    fused_volume_path = os.path.join(save_folder, 'fused_volume.npy')
    np.save(fused_volume_path, fused_volume)
    print(f"Fused volume saved to {fused_volume_path}")


if __name__ == "__main__":
    save_folder = '/home/thales1/VSLAM/Vslam_algorithm/pythonProject1/Visual_slam_optimized/Dynamic_Static_scenes/volumes'  # Specify the folder where the volumes are saved
    volume_fusion_process(save_folder)
