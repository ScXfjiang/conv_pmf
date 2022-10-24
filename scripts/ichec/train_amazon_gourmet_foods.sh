#!/bin/bash -l
#SBATCH --job-name=conv_pmf
# speficity number of nodes
#SBATCH -N 1
# specify the gpu queue
#SBATCH --partition=GpuQ
# specify number of tasks/cores per node required
#SBATCH --ntasks-per-node=35
# specify the walltime e.g 96 hours
#SBATCH -t 96:00:00
# set to email at start,end and failed jobs
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sc.xfjiang@gmail.com

# run from current directory
cd $SLURM_SUBMIT_DIR

module load cuda/11.3

DATA_PATH="/ichec/work/ucd01/xfjiang/amazon/amazon_grocery_and_gourmet_foods1"

for idx in 1 2 3; do
    for cuda_device_idx in 0 1; do
        export CUDA_VISIBLE_DEVICES=${cuda_device_idx}
        python ../../src/train.py \
            --dataset_type="amazon_grocery_and_gourmet_foods" \
            --train_dataset_path="${DATA_PATH}/train.json" \
            --val_dataset_path="${DATA_PATH}/val.json" \
            --val_dataset_path="${DATA_PATH}/test.json" \
            --word_embeds_type="glove.6B.50d" \
            --word_embeds_path="/ichec/work/ucd01/xfjiang/glove.6B/glove.6B.50d.txt" \
            --global_user_id2global_user_idx="${DATA_PATH}/global_user_id2global_user_idx.pkl" \
            --global_item_id2global_item_idx="${DATA_PATH}/global_item_id2global_item_idx.pkl" \
            --shuffle=True \
            --train_batch_size=256 \
            --val_batch_size=256 \
            --num_epoch=60 \
            --window_size=5 \
            --n_word=128 \
            --n_factor=32 \
            --with_entropy=True \
            --epsilon=1e-5 \
            --lr=0.1 \
            --momentum=0.9 \
            --weight_decay=0.0001 \
            --use_cuda=True &
    done
    for JOB in $(jobs -p); do
        wait ${JOB}
    done
done
