#!/bin/sh
IMG_DIR=/home/zhangkai/data/allweather/gt
OUT_DIR=/home/zhangkai/data/allweather/depth_results/gt

python3 run.py --encoder vits --img-path ${IMG_DIR} --outdir ${OUT_DIR}