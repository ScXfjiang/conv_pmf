#!/bin/bash
# this script should be run on sonic and then copy the output to other servers
ORIGINAL_DATASET_PATH="/scratch/22200056/dataset/amazon/reviews_Grocery_and_Gourmet_Food_5.json"
OUTPUT_DIR="/home/people/22200056/workspace/preprocess_output"
REF_CORPUS="/scratch/22200056/dataset/wikitext/train.metadata.jsonl"
WORD_EMBEDDING_PATH="/scratch/22200056/dataset/glove.6B/glove.6B.50d.txt"

python ../src/preprocess.py \
    --src=${ORIGINAL_DATASET_PATH} \
    --clean_corpus="T" \
    --dst=${OUTPUT_DIR} \
    --reference=${REF_CORPUS} \
    --word_embeds=${WORD_EMBEDDING_PATH}
