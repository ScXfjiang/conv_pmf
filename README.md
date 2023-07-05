# conv_pmf

This repository is the implementation of [Enhancing Topic Extraction in Recommender Systems with Entropy Regularization](https://arxiv.org/abs/2306.07403).

## Reproduce process:
1. Download Amazon Dataset: http://jmcauley.ucsd.edu/data/amazon/links.html. The statistics of all datasets are illustrated in this repo: https://github.com/ScXfjiang/dataset_insight. We suggest you use **Grocery and Gourmet Foods** as in the paper because it has reasonable number of users and items.
2. Preprocess.
```
python ../src/preprocess.py \
    --src=${ORIGINAL_DATASET_PATH} \
    --clean_corpus="T" \
    --dst=${OUTPUT_DIR} \
    --reference=${ORIGINAL_DATASET_PATH} \
    --word_embeds=${WORD_EMBEDDING_PATH}
```
3. Training and Evaluation 
```
python ../src/run.py \
    --dataset_path="${DATA_PATH}" \
    --word_embeds_path=${WORD_EMBEDDING_PATH} \
    --global_user_id2global_user_idx="${DATA_PATH}/global_user_id2global_user_idx.pkl" \
    --global_item_id2global_item_idx="${DATA_PATH}/global_item_id2global_item_idx.pkl" \
    --shuffle=True \
    --train_batch_size=256 \
    --val_batch_size=256 \
    --num_epoch=35 \
    --window_size=5 \
    --n_word=64 \
    --n_factor=${n_factor} \
    --epsilon=${epsilon} \
    --lr=0.1 \
    --momentum=0.9 \
    --weight_decay=0.0001 \
    --ew_batch_size=1024 \
    --ew_least_act_num=20 \
    --ew_k=10 \
    --ew_token_cnt_mat_path="${DATA_PATH}/token_cnt_mat.npz" \
    --log_dir="n_factor_${n_factor}" \
    --log_dir_level_2="${epsilon}"
```
4. Extract Topic Keywords
```
python ../src/extract_words.py \
    --dataset_path="${DATA_PATH}" \
    --word_embeds_path=${WORD_EMBEDDING_PATH} \
    --checkpoint_path="${CHECKPOINT}" \
    --n_factor=${n_factor} \
    --n_word=64 \
    --window_size=5 \
    --strategy="all" \
    --batch_size=1024 \
    --least_act_num=20 \
    --k=10 \
    --log_dir_level_1="n_factor_${n_factor}" \
    --log_dir_level_2="${epsilon}"
```

## Experiment Results
Comparison of topic coherence (word embedding cosine similarity) for different entropy regularization coefficients ($\lambda$ s), with $\lambda=0.0$ indicating the absence of entropy regularization terms. The highest-performing result is highlighted in bold.
| $n$\_factor | $0.0$ | $0.4$ | $0.8$ | $1.2$ | $1.6$ | $2.0$ |
|-------------|-------|-------|-------|-------|-------|-------|
| $6$         | 0.2850| 0.2932| 0.3508| 0.3863| 0.4043| **0.4161**|
| $8$         | 0.2737| 0.2825| 0.3485| 0.3619| 0.3964| **0.4112**|
| $10$        | 0.2634| 0.283 | 0.3132| 0.3582| 0.3743| **0.3982**|
| $12$        | 0.2424| 0.2804| 0.2937| 0.3599| 0.3722| **0.3808**|

Comparison of rating prediction accuracy (RMSE) between base models and ConvMF under different entropy regularization coefficients ($\lambda$ s), with $\lambda=0.0$ indicating the absence of entropy regularization terms. The highest-performing result is highlighted in bold.
| $n$\_factor | Offset | PMF | $0.0$ | $0.4$ | $0.8$ | $1.2$ | $1.6$ | $2.0$ |
|-------------|--------|-----|-------|-------|-------|-------|-------|-------|
| $6$         | 1.1722 |1.1467| **1.0632**| 1.0765| 1.0920| 1.0927| 1.0927| 1.0940|
| $8$         | 1.1722 |1.1559| **1.0662**| 1.0797| 1.0915| 1.0902| 1.0982| 1.1047|
| $10$        | 1.1722 |1.1607| **1.0735**| 1.0805| 1.0895| 1.0945| 1.0985| 1.1010|
| $12$        | 1.1722 |1.1661| **1.0778**| 1.0852| 1.0902| 1.0985| 1.10325|1.1117|

Illustration of extracted topic keywords. Here, the total number of latent factors is fixed at 8. For each latent factor, we compute the word2vec cosine similarities, which are displayed at the top-left corner of each corresponding word cloud. The topics are sorted by this metric in descending order. Additionally, we calculate the average word2vec cosine similarities for all latent factors, yielding values of 0.2781 and 0.4270 respectively.
<img width="1431" alt="keywords" src="https://github.com/ScXfjiang/conv_pmf/assets/13879402/7ef1fd22-1f10-4552-bc59-232eff3f5a5f">

