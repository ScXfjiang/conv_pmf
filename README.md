# conv_pmf

Welcome to the repository for the implementation of "[Enhancing Topic Extraction in Recommender Systems with Entropy Regularization](https://arxiv.org/abs/2306.07403)".

## Steps to Reproduce the Process:
1. **Data Acquisition**: Download the Amazon Dataset from [here](http://jmcauley.ucsd.edu/data/amazon/links.html). This [repository](https://github.com/ScXfjiang/dataset_insight) provides statistics for all the datasets. We recommend using the **Grocery and Gourmet Foods** dataset for its reasonable number of users and items, as done in the paper.

2. **Data Preprocessing**: Execute the following command.
```bash
python ../src/preprocess.py \
    --src=${ORIGINAL_DATASET_PATH} \
    --clean_corpus="T" \
    --dst=${OUTPUT_DIR} \
    --reference=${ORIGINAL_DATASET_PATH} \
    --word_embeds=${WORD_EMBEDDING_PATH}
```
3. **Model Training and Evaluation**: Run the following command.
```bash
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
4. **Topic Keyword Extraction**: Use the following command.
```bash
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
The experimental results presented in this section evaluate both topic coherence (based on NPMI and word embedding cosine similarity) and rating prediction accuracy (root mean square error - RMSE). We vary entropy regularization coefficients ($\lambda$ s) for these comparisons, with $\lambda=0.0$ symbolizing no entropy regularization terms. The best-performing results are highlighted in bold.

### Topic Coherence Comparison
#### NPMI
|  n\_factor  |  0.0   |  0.4   |  0.8   |  1.2   |  1.6   |  2.0   |
| ----------- | :----: | :----: | :----: | :----: | :----: | :----: |
| $6$  | 0.0837 | 0.0947 | 0.1754 | 0.1945 | 0.2038 | **0.2031** |
| $8$  | 0.0820 | 0.0911 | 0.1519 | 0.1645 | 0.1683 | **0.1930** |
| $10$ | 0.0796 | 0.0871 | 0.1221 | 0.1602 | 0.1674 | **0.1701** |
| $12$ | 0.0621 | 0.0713 | 0.0919 | 0.1483 | 0.1515 | **0.1736** |

#### W2V Similarity
| n\_factor | 0.0 | 0.4 | 0.8 | 1.2 | 1.6 | 2.0 |
|-------------|-------|-------|-------|-------|-------|-------|
| $6$         | 0.2850| 0.2932| 0.3508| 0.3863| 0.4043| **0.4161**|
| $8$         | 0.2737| 0.2825| 0.3485| 0.3619| 0.3964| **0.4112**|
| $10$        | 0.2634| 0.283 | 0.3132| 0.3582| 0.3743| **0.3982**|
| $12$        | 0.2424| 0.2804| 0.2937| 0.3599| 0.3722| **0.3808**|

### RMSE Comparison
| n\_factor | Offset | PMF | 0.0 | 0.4 | 0.8 | 1.2 | 1.6 | 2.0 |
|-------------|--------|-----|-------|-------|-------|-------|-------|-------|
| $6$         | 1.1722 |1.1467| **1.0632**| 1.0765| 1.0920| 1.0927| 1.0927| 1.0940|
| $8$         | 1.1722 |1.1559| **1.0662**| 1.0797| 1.0915| 1.0902| 1.0982| 1.1047|
| $10$        | 1.1722 |1.1607| **1.0735**| 1.0805| 1.0895| 1.0945| 1.0985| 1.1010|
| $12$        | 1.1722 |1.1661| **1.0778**| 1.0852| 1.0902| 1.0985| 1.10325|1.1117|

In the example below, the total count of latent factors is set at 8. For each latent factor, we compute the word2vec cosine similarities, which are depicted at the top-left corner of each corresponding word cloud. These topics are arranged in descending order based on the cosine similarity. We also compute the average word2vec cosine similarities for all latent factors, yielding values of 0.2781 and 0.4270 respectively.
<img width="1431" alt="keywords" src="https://github.com/ScXfjiang/conv_pmf/assets/13879402/7ef1fd22-1f10-4552-bc59-232eff3f5a5f">

