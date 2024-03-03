#!/bin/sh

script_dir=$(realpath $(dirname $0))

IMG_DIR=/home/user/Depth-Anything/visualize_features/input/439_rain.png
OUT_DIR=$script_dir/mem_plot

python3 get_dpt_mem.py --encoder vitl14 --resizeh 518 --img-path ${IMG_DIR} --outdir ${OUT_DIR}