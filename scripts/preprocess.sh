#!/bin/bash
# this script should be run on sonic and then copy the output to other servers
python ../src/preprocess.py \
    --src="/scratch/22200056/dataset/amazon/reviews_Grocery_and_Gourmet_Food_5.json" \
    --clean_corpus="T" \
    --dst="/scratch/22200056/dataset/amazon/amazon_grocery_and_gourmet_foods_clean" \
    --reference="/scratch/22200056/dataset/amazon/reviews_Grocery_and_Gourmet_Food_5.json" \
    --word_embeds="/scratch/22200056/dataset/glove.6B/glove.6B.50d.txt"
