import os
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
import cv2

class MergePlots(object):
    def __init__(self, args) -> None:
        self.image_dir = Path(args.imgdir)
        self.image_fp = list(self.image_dir.glob("*.*"))
        self.image_fp = [str(i) for i in self.image_fp]

        self.data_dir = Path(args.datadir)
        self.data_fp = list(self.data_dir.glob("*.npy"))
        self.data_fp = [str(i) for i in self.data_fp]

        self.outdir = args.outdir
        Path.mkdir(Path(self.outdir).resolve(), exist_ok=True)

        self.data_dict = {}

        for imgfp in self.image_fp:
            img_name = str(imgfp).split(os.path.sep)[-1]


        for key in self.data_dict.keys():
            self.data_dict[key].sort()

    def plot_features(self):
        for imgfp in self.image_fp:
            img_name = str(imgfp).split(os.path.sep)[-1]
            data_dict = {
                "dpt": None,
                "dino1": None,
                "dinov2": None,
                "vgg16": None
            }
            # Find corresponding feature data
            for dfp in self.data_fp:
                if img_name in dfp:
                    for key in data_dict.keys():
                        if key in dfp:
                            data_dict[key] = dfp
            
            self.merge_plot(imgfp, data_dict)

    def merge_plot(self, img_fp, data_dict):
        print("img fp: {}".format(img_fp))
        print("data dict: {}".format(data_dict))
        img_name = str(img_fp).split(os.path.sep)[-1]
        fig, axes = plt.subplots(1, 5, figsize=(12, 5))

        raw_image = cv2.imread(img_fp)
        image_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)

        axes[0].imshow(image_pil)
        axes[0].axis('off')  
        axes[0].set_title('Original Image')

        dpt_features = np.load(data_dict["dpt"])
        feature_image = axes[1].imshow(dpt_features, cmap='viridis', interpolation='nearest')
        axes[1].axis('off')
        axes[1].set_title('DepAny Feature Map')

        vgg_features = np.load(data_dict["vgg16"])
        feature_image = axes[2].imshow(vgg_features, cmap='viridis', interpolation='nearest')
        axes[2].axis('off')
        axes[2].set_title('VGG16 Feature Map')

        dino_features = np.load(data_dict["dino1"])
        feature_image = axes[3].imshow(dino_features, cmap='viridis', interpolation='nearest')
        axes[3].axis('off')
        axes[3].set_title('Dino Feature Map')

        dinov2_features = np.load(data_dict["dinov2"])
        feature_image = axes[4].imshow(dinov2_features, cmap='viridis', interpolation='nearest')
        axes[4].axis('off')
        axes[4].set_title('Dinov2 Feature Map')

        fig.tight_layout()
        fig.colorbar(feature_image, ax=axes.ravel().tolist(), shrink=0.75)
        save_path = str(Path(self.outdir).joinpath(img_name))
        plt.savefig(save_path, bbox_inches='tight')
        pass
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgdir', type=str)
    parser.add_argument('--datadir', type=str)
    parser.add_argument('--outdir', type=str)
    mp = MergePlots(parser.parse_args())
    mp.plot_features()