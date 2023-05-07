#!/bin/bash -l
#SBATCH --job-name=maskrcnn
# speficity number of nodes
#SBATCH -N 1
# specify the gpu queue
#SBATCH --partition=csgpu
# Request 2 gpus
#SBATCH --gres=gpu:2
# specify number of tasks/cores per node required
#SBATCH --ntasks-per-node=35
# specify the walltime e.g 192 hours
#SBATCH -t 192:00:00
# set to email at start,end and failed jobs
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sc.xfjiang@gmail.com

# run from current directory
cd $SLURM_SUBMIT_DIR

DATA_PATH="/scratch/22204923/datasets/amazon/amazon_grocery_and_gourmet_foods_clean"

for n_factor in 8; do
    for epsilon in 0.0 0.01 0.05 0.1 0.5 1.0 3.0; do
        for cuda_device_idx in 0 1; do
            export CUDA_VISIBLE_DEVICES=${cuda_device_idx}
            python ../src/run.py \
                --dataset_path="${DATA_PATH}" \
                --word_embeds_path="/scratch/22204923/datasets/glove.6B/glove.6B.50d.txt" \
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
                --ew_least_act_num=10 \
                --ew_k=10 \
                --ew_token_cnt_mat_path="${DATA_PATH}/token_cnt_mat.npz" \
                --log_dir="n_factor_${n_factor}" \
                --log_dir_level_2="${epsilon}" &
        done
        for JOB in $(jobs -p); do
            wait ${JOB}
        done
    done
done

