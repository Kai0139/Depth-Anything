import argparse
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm

from pathlib import Path

from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

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
    
    # depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(args.encoder)).to(DEVICE).eval()
    depth_anything = DepthAnything.from_pretrained('/home/zhangkai/repos/Depth-Anything/models/vits14').to(DEVICE).eval()
    
    total_params = sum(param.numel() for param in depth_anything.parameters())
    print('Total parameters: {:.2f}M'.format(total_params / 1e6))
    
    transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
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

    print("dinov2 chunked blocks: {}".format(depth_anything.pretrained.chunked_blocks))
    
    for filename in tqdm(filenames):
        print(filename)
        fp = Path(filename)
        image_fn = str(fp).split("/")[-1]
        feature_image_path = str(Path(__file__).resolve().parent.joinpath("feature_images", image_fn))

        raw_image = cv2.imread(filename)
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        
        h, w = image.shape[:2]
        print("image h: {} w: {}".format(h, w))        
        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)
        print("input size: {}".format(image.shape))
        
        with torch.no_grad():
            features = depth_anything(image, True)
            last_feature_block = features[3][0] # 1x 2072 x 384
            last_feature_block = torch.squeeze(last_feature_block, 0) # 2072 x 384
            
            ch = last_feature_block.size()[1] # ch = 384
            print("number of channels: {}".format(ch))
            # visualize all 384 channels with 37 * 56
            seq_h = int(image.shape[2] / 14)
            seq_w = int(image.shape[3] / 14)
            feature_channels = []
            for i in range(ch):
                feature_ch = last_feature_block[:,i]
                feature_ch = feature_ch.reshape(seq_h, seq_w).cpu().numpy()
                # cv2.resize(feature_ch, (seq_h*2, seq_w*2), cv2.INTER_NEAREST)
                feature_channels.append(feature_ch)
                 
            viz_ncols = 24
            viz_w = viz_ncols * seq_w
            viz_nrows = int(np.ceil(len(feature_channels) / viz_ncols))
            print("number of rows in viz image: {}".format(viz_nrows))
            viz_rows = []
            for i in range(viz_nrows-1):
                head_idx = i * viz_ncols
                tail_idx = np.min([i*viz_ncols + viz_ncols, len(feature_channels)])
                # print("head idx of row: {}".format(head_idx))
                # print("tail idx of row: {}".format(tail_idx))
                viz_row = cv2.hconcat(feature_channels[head_idx: tail_idx])
                viz_rows.append(viz_row)

            print("single row shape: {}".format(viz_rows[0].shape))
            viz = cv2.vconcat(viz_rows)
            last_row = cv2.hconcat(feature_channels[(viz_nrows-1)*viz_ncols: len(feature_channels)])
            if last_row.shape[1] < viz_w:
                remainders = np.zeros([seq_h, viz_w - last_row.shape[1]], dtype=int)
                last_row = cv2.hconcat([last_row, remainders])
            viz = cv2.vconcat([viz, last_row])
            print("final viz image shape: {}".format(viz.shape))
            viz = normalize_255(viz)
            viz = cv2.cvtColor(viz, cv2.COLOR_GRAY2RGB)
            
            # Put original image in the viz image
            image_h = raw_image.shape[0]
            image_w = raw_image.shape[1]
            target_h = np.max([image_h + 2*margin_width, viz.shape[0] + 2*margin_width])
            feature_h = viz.shape[0]
            
            if (target_h - image_h) % 2 != 0:
                img_margin_top = int((target_h - image_h) / 2)
                img_margin_bot = int((target_h - image_h) / 2) + 1
            else:
                img_margin_top = int((target_h - image_h) / 2)
                img_margin_bot = int((target_h - image_h) / 2)

            if (target_h - feature_h) % 2 != 0:
                feature_margin_top = int((target_h - feature_h) / 2)
                feature_margin_bot = int((target_h - feature_h) / 2) + 1
            else:
                feature_margin_top = int((target_h - feature_h) / 2)
                feature_margin_bot = int((target_h - feature_h) / 2)

            img_left = cv2.vconcat([np.ones([img_margin_top, image_w, 3], dtype=np.uint8), 
                                    raw_image, 
                                    np.ones([img_margin_bot, image_w, 3], dtype=np.uint8)])
            img_right = cv2.vconcat([np.ones([feature_margin_top, viz.shape[1], 3], dtype=np.uint8), 
                                    viz, 
                                    np.ones([feature_margin_bot, viz.shape[1], 3], dtype=np.uint8)])
            viz = cv2.hconcat([img_left, img_right])
            cv2.imwrite(feature_image_path, viz)
