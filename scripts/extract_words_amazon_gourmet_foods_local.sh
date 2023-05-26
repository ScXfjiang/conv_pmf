DATA_PATH="/home/ubuntu/dataset/amazon_grocery_and_gourmet_foods_clean"
CHECKPOINT_PATH="/home/ubuntu/dataset/checkpoint_final.pt"

python ../src/extract_words.py \
    --dataset_path="${DATA_PATH}" \
    --word_embeds_path="/home/ubuntu/dataset/glove.6B/glove.6B.50d.txt" \
    --checkpoint_path="${CHECKPOINT_PATH}" \
    --n_factor=8 \
    --n_word=64 \
    --window_size=5 \
    --batch_size=1024 \
    --least_act_num=20 \
    --k=10 
