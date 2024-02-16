from pathlib import Path
import random
import shutil

if __name__ == "__main__":
    input_images = list(Path("/data/kaizhang/allweather/input").glob("*.*"))
    selected_images = random.sample(input_images, 100)
    for src_img in selected_images:
        print(str(src_img))
        shutil.copy(str(src_img), "/home/user/Depth-Anything/visualize_features/input_rand/")