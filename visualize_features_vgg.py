import argparse
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from torchvision.models.vgg import VGG

os.environ['CUDA_VISIBLE_DEVICES'] = '6'

def normalize_255(arr : np.array):
    min_x = np.min(arr)
    max_x = np.max(arr)
    result = 255 * (arr - min_x) / (max_x - min_x)
    result = result.astype(np.uint8)
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-path', type=str)
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'])
    
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    
    args = parser.parse_args()
    
    margin_width = 50
    caption_height = 60
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # vits16 = torch.hub.load('facebookresearch/dino:main', 'dino_vits16').to(DEVICE).eval()
    vgg16 = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16').to(DEVICE).eval()
    
    total_params = sum(param.numel() for param in vgg16.parameters())
    print('Total parameters: {:.2f}M'.format(total_params / 1e6))
    
    transform = Compose([
        Resize(
            width=518*4,
            height=518*4,
            resize_target=False,
            keep_aspect_ratio=True,
            # ensure_multiple_of=16,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])
    
    if os.path.isfile(args.img_path):
        if args.img_path.endswith('txt'):
            with open(args.img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]
    else:
        filenames = os.listdir(args.img_path)
        filenames = [os.path.join(args.img_path, filename) for filename in filenames if not filename.startswith('.')]
        filenames.sort()
    
    os.makedirs(args.outdir, exist_ok=True)
    
    for filename in tqdm(filenames):
        print(filename)
        fp = Path(filename)
        image_fn = str(fp).split("/")[-1]
        feature_image_path = str(Path(__file__).resolve().parent.joinpath("dino1_feature_images", image_fn))

        raw_image = cv2.imread(filename)
        image_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)

        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        
        h, w = image.shape[:2]
        print("image h: {} w: {}".format(h, w))        
        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)
        print("input size: {}".format(image.shape))
        
        with torch.no_grad():
            features = vgg16.features(image)
            print("features shape: {}".format(features.shape))
            last_feature_block = features.squeeze(0).permute(1,2,0)
            last_feature_block = torch.mean(last_feature_block, dim=2)
            print("last_feature_block shape: {}".format(last_feature_block.shape))

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            axes[0].imshow(image_pil)
            axes[0].axis('off')  
            axes[0].set_title('Original Image')

            feature_image = axes[1].imshow(last_feature_block.cpu().numpy(), cmap='viridis', interpolation='nearest')
            axes[1].axis('off')
            axes[1].set_title('Feature Map')

            fig.colorbar(feature_image, ax=axes.ravel().tolist(), shrink=0.75)
            plt.savefig(feature_image_path, bbox_inches='tight')
