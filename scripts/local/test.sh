#!/home/xuefei/anaconda3/bin/zsh
DATA_PATH="/data/xuefei/amazon/amazon_grocery_and_gourmet_foods1"
CHECKPOINT_PATH=$1
N_FACTOR=$2

checkpoint_files=()
for ENTRY in "${CHECKPOINT_PATH}"/*; do
    checkpoint_files+="${ENTRY}/checkpoint/checkpoint_50.pt"
done
for checkpoint in ${checkpoint_files}; do
    python ../../src/test.py \
        --dataset_type="amazon_grocery_and_gourmet_foods" \
        --test_dataset_path="${DATA_PATH}/test.json" \
        --word_embeds_type="glove.6B.50d" \
        --word_embeds_path="/data/xuefei/glove.6B/glove.6B.50d.txt" \
        --global_user_id2global_user_idx="${DATA_PATH}/global_user_id2global_user_idx.pkl" \
        --global_item_id2global_item_idx="${DATA_PATH}/global_item_id2global_item_idx.pkl" \
        --checkpoint_path=${checkpoint} \
        --test_batch_size=256 \
        --window_size=5 \
        --n_word=128 \
        --n_factor=${N_FACTOR} \
        --use_cuda=True
done
