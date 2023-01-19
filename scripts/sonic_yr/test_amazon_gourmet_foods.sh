#!/bin/bash -l
#SBATCH --job-name=maskrcnn_test
# speficity number of nodes
#SBATCH -N 1
# specify the gpu queue
#SBATCH --partition=csgpu
# Request 2 gpus
#SBATCH --gres=gpu:2
# specify number of tasks/cores per node required
#SBATCH --ntasks-per-node=35
# specify the walltime e.g 96 hours
#SBATCH -t 96:00:00

# run from current directory
cd $SLURM_SUBMIT_DIR

DATA_PATH="/home/people/22200056/workspace/dataset/amazon/amazon_grocery_and_gourmet_foods1"

for LOG_DIR in Sep-10-2022-17-28-52 Sep-10-2022-17-56-25 Sep-11-2022-12-40-31 Sep-11-2022-15-57-25 Sep-11-2022-19-16-36; do
    python ../../src/test.py \
        --dataset_type="amazon_grocery_and_gourmet_foods" \
        --train_dataset_path="${DATA_PATH}/train.json" \
        --val_dataset_path="${DATA_PATH}/val.json" \
        --test_dataset_path="${DATA_PATH}/test.json" \
        --word_embeds_path="/home/people/22200056/workspace/dataset/glove.6B/glove.6B.50d.txt" \
        --global_user_id2global_user_idx="${DATA_PATH}/global_user_id2global_user_idx.pkl" \
        --global_item_id2global_item_idx="${DATA_PATH}/global_item_id2global_item_idx.pkl" \
        --checkpoint_path="/scratch/22200056/conv_pmf_results/conv_pmf_entropy_A/epsilon_0.005/${LOG_DIR}/checkpoint/checkpoint_50.pt" \
        --test_batch_size=256 \
        --window_size=5 \
        --n_word=128 \
        --n_factor=32
done
