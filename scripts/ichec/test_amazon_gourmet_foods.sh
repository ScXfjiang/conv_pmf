#!/bin/bash -l
#SBATCH --job-name=conv_pmf_test
# speficity number of nodes
#SBATCH -N 1
# specify the gpu queue
#SBATCH --partition=GpuQ
# specify number of tasks/cores per node required
#SBATCH --ntasks-per-node=35
# specify the walltime e.g 48 hours
#SBATCH -t 48:00:00

# run from current directory
cd $SLURM_SUBMIT_DIR

module load cuda/11.3

DATA_PATH="/ichec/work/ucd01/xfjiang/amazon/amazon_grocery_and_gourmet_foods1"

for LOG_DIR in Sep-10-2022-14-34-44; do
    python ../../src/test.py \
        --dataset_type="amazon_grocery_and_gourmet_foods" \
        --test_dataset_path="${DATA_PATH}/train.json" \
        --test_dataset_path="${DATA_PATH}/val.json" \
        --test_dataset_path="${DATA_PATH}/test.json" \
        --word_embeds_type="glove.6B.50d" \
        --word_embeds_path="/ichec/work/ucd01/xfjiang/glove.6B/glove.6B.50d.txt" \
        --global_user_id2global_user_idx="${DATA_PATH}/global_user_id2global_user_idx.pkl" \
        --global_item_id2global_item_idx="${DATA_PATH}/global_item_id2global_item_idx.pkl" \
        --checkpoint_path="/ichec/work/ucd01/xfjiang/experiments/conv_pmf_with_entropy_A/scripts/amazon_grocery_and_gourmet_foods/${LOG_DIR}/checkpoint/checkpoint_50.pt" \
        --test_batch_size=256 \
        --window_size=5 \
        --n_word=128 \
        --n_factor=32 \
        --use_cuda=True
done
