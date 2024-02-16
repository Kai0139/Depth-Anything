#!/bin/sh
IMG_DIR=/home/user/Depth-Anything/input
OUT_DIR=/home/user/Depth-Anything/visualize_features/data

python3 visualize_features_dpt.py --encoder vits --img-path ${IMG_DIR} --outdir ${OUT_DIR}