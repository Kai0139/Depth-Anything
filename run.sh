#!/bin/sh
IMG_DIR=/home/user/Depth-Anything/input
OUT_DIR=/home/kaizhang/data/mini_allweather/feature_viz/input

python3 visualize_features_all.py --encoder vits --img-path ${IMG_DIR} --outdir ${OUT_DIR}