#!/bin/bash
DATA_PATH="/data/xuefei/dataset/amazon/amazon_grocery_and_gourmet_foods"

python ../../src/train.py \
    --dataset_path="${DATA_PATH}" \
    --word_embeds_path="/data/xuefei/dataset/glove.6B/glove.6B.50d.txt" \
    --global_user_id2global_user_idx="${DATA_PATH}/global_user_id2global_user_idx.pkl" \
    --global_item_id2global_item_idx="${DATA_PATH}/global_item_id2global_item_idx.pkl" \
    --shuffle=True \
    --train_batch_size=256 \
    --val_batch_size=256 \
    --num_epoch=35 \
    --window_size=5 \
    --n_word=16 \
    --n_factor=8 \
    --epsilon=0.0 \
    --lr=0.1 \
    --momentum=0.9 \
    --weight_decay=0.0001 \
    --ew_batch_size=1024 \
    --ew_least_act_num=30 \
    --ew_k=10 \
    --ew_token_cnt_mat_path="${DATA_PATH}/token_cnt_mat.npz" \
    --log_dir="n_factor_8" \
    --log_dir_level_2="0.0"
