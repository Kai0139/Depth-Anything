from pathlib import Path
import cv2
import numpy as np

if __name__ == '__main__':
    root_path = Path("/mnt/slurmfs-4090node1/user_data/kzhang740/data/glass_det/trans10k/")

    img_path = root_path.joinpath("validation", "hard", "images", "2926.jpg")
    depth_path = root_path.joinpath("validation", "hard", "dam", "2926.npy")

    img = cv2.imread(str(img_path))
    depth = np.load(str(depth_path))
    
    depth = (depth / np.max(depth)) * 255
    depth = depth.astype(np.uint8)
    # depth = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
    depth = cv2.resize(depth, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    print(depth[920, 4000])
    cv2.imwrite("depth.png", depth)
