#!/bin/sh

script_dir=$(realpath $(dirname $0))

IMG_DIR=/home/user/Depth-Anything/input
OUT_DIR=$script_dir/data

DATA_DIR=${OUT_DIR}
RESULT_DIR=$script_dir/results

# python3 visualize_features_dinov2.py --encoder vits --resizeh 1554 --img-path ${IMG_DIR} --outdir ${OUT_DIR}
# python3 visualize_features_dino1.py --encoder vits --resizeh 1554 --img-path ${IMG_DIR} --outdir ${OUT_DIR}
# python3 visualize_features_dpt.py --encoder vits --resizeh 1554 --img-path ${IMG_DIR} --outdir ${OUT_DIR}
# python3 visualize_features_vgg.py --encoder vits --resizeh 2072 --img-path ${IMG_DIR} --outdir ${OUT_DIR}

python3 merge_plots.py --imgdir ${IMG_DIR} --datadir ${DATA_DIR} --outdir ${RESULT_DIR}