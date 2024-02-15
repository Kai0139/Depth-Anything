#!/bin/sh
IMG_DIR=/home/zhangkai/data/mini_allweather/input
OUT_DIR=/home/zhangkai/data/mini_allweather/feature_viz/input

python3 visualize_features_dino1.py --encoder vits --img-path ${IMG_DIR} --outdir ${OUT_DIR}