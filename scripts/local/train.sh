#!/bin/bash

DATA_PATH="/data/xuefei/dataset/amazon/amazon_grocery_and_gourmet_foods1"

python ../../src/train.py \
    --dataset_path="${DATA_PATH}" \
    --word_embeds_path="/data/xuefei/dataset/glove.6B/glove.6B.50d.txt" \
    --global_user_id2global_user_idx="${DATA_PATH}/global_user_id2global_user_idx.pkl" \
    --global_item_id2global_item_idx="${DATA_PATH}/global_item_id2global_item_idx.pkl" \
    --shuffle=True \
    --train_batch_size=256 \
    --val_batch_size=256 \
    --num_epoch=60 \
    --window_size=5 \
    --n_word=128 \
    --n_factor=8 \
    --with_entropy=False \
    --epsilon=1e-4 \
    --lr=0.1 \
    --momentum=0.9 \
    --weight_decay=0.0001
