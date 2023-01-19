#!/bin/bash -l
#SBATCH --job-name=wide&deep
# speficity number of nodes
#SBATCH -N 1
# specify the gpu queue
#SBATCH --partition=GpuQ
# specify number of tasks/cores per node required
#SBATCH --ntasks-per-node=35
# specify the walltime e.g 48 hours
#SBATCH -t 48:00:00
# set to email at start,end and failed jobs
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sc.xfjiang@gmail.com

# run from current directory
cd $SLURM_SUBMIT_DIR

module load cuda/11.3

DATA_PATH="/ichec/work/ucd01/yongru/dataset/amazon/amazon_grocery_and_gourmet_foods"

for idx in 1; do
    for cuda_device_idx in 0 1; do
        export CUDA_VISIBLE_DEVICES=${cuda_device_idx}
        python ../../src/train.py \
            --dataset_path="${DATA_PATH}" \
            --word_embeds_path="/ichec/work/ucd01/yongru/dataset/glove.6B/glove.6B.50d.txt" \
            --global_user_id2global_user_idx="${DATA_PATH}/global_user_id2global_user_idx.pkl" \
            --global_item_id2global_item_idx="${DATA_PATH}/global_item_id2global_item_idx.pkl" \
            --shuffle=True \
            --train_batch_size=256 \
            --val_batch_size=256 \
            --num_epoch=60 \
            --window_size=5 \
            --n_word=128 \
            --n_factor=32 \
            --epsilon=0.0 \
            --lr=0.1 \
            --momentum=0.9 \
            --weight_decay=0.0001 &
    done
    for JOB in $(jobs -p); do
        wait ${JOB}
    done
done
