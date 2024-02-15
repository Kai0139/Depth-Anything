#!/bin/sh
IMG_DIR=/home/user/Depth-Anything/input
OUT_DIR=/home/user/Depth-Anything/combined_features

python3 visualize_features_all.py --encoder vits --img-path ${IMG_DIR} --outdir ${OUT_DIR}