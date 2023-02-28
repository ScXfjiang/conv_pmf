#!/bin/bash -l
#SBATCH --job-name=conv_pmf
# speficity number of nodes
#SBATCH -N 1
# specify the gpu queue
#SBATCH --partition=csgpu
# Request 2 gpus
#SBATCH --gres=gpu:2
# specify number of tasks/cores per node required
#SBATCH --ntasks-per-node=35
# specify the walltime e.g 96 hours
#SBATCH -t 192:00:00
# set to email at start,end and failed jobs
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sc.xfjiang@gmail.com

# run from current directory
cd $SLURM_SUBMIT_DIR

DATA_PATH="/home/people/22200056/scratch/dataset/amazon/amazon_grocery_and_gourmet_foods"

for n_factor in 4 8 16 32 64 128; do
    for epsilon in 10 0.8 0.6 0.4 0.2 0.08 0.06 0.04 0.02; do
        for idx in 1; do
            for cuda_device_idx in 0 1; do
                export CUDA_VISIBLE_DEVICES=${cuda_device_idx}
                python ../../src/train.py \
                    --dataset_path="${DATA_PATH}" \
                    --word_embeds_path="/scratch/22200056/dataset/glove.6B/glove.6B.50d.txt" \
                    --global_user_id2global_user_idx="${DATA_PATH}/global_user_id2global_user_idx.pkl" \
                    --global_item_id2global_item_idx="${DATA_PATH}/global_item_id2global_item_idx.pkl" \
                    --shuffle=True \
                    --train_batch_size=256 \
                    --num_epoch=25 \
                    --window_size=5 \
                    --n_word=16 \
                    --n_factor=${n_factor} \
                    --epsilon=${epsilon} \
                    --lr=0.1 \
                    --momentum=0.9 \
                    --weight_decay=0.0001 \
                    --log_dir="n_factor_${n_factor}" \
                    --log_dir_level_2="${epsilon}" &
            done
            for JOB in $(jobs -p); do
                wait ${JOB}
            done
        done
    done
done
