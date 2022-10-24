#!/bin/bash -l
#SBATCH --job-name=extract_words
# speficity number of nodes
#SBATCH -N 1
# specify the gpu queue
#SBATCH --partition=GpuQ
# specify number of tasks/cores per node required
#SBATCH --ntasks-per-node=35
# specify the walltime e.g 96 hours
#SBATCH -t 96:00:00

# run from current directory
cd $SLURM_SUBMIT_DIR

module load cuda/11.3

DATA_PATH="/ichec/work/ucd01/xfjiang/amazon/amazon_grocery_and_gourmet_foods1"
CHECKPOINT_PATH="/ichec/work/ucd01/xfjiang/checkpoints/conv_pmf_checkpoints"

for ENTROPY_THRESHOLD in 4.84 4.83 4.82 4.81 4.80 4.75 4.70 4.60 4.50; do
    for LOG_DIR in Sep-03-2022-21-06-38; do
        python ../../src/extract_words.py \
            --dataset_type="amazon_grocery_and_gourmet_foods" \
            --train_dataset_path="${DATA_PATH}/train.json" \
            --word_embeds_type="glove.6B.50d" \
            --word_embeds_path="/ichec/work/ucd01/xfjiang/glove.6B/glove.6B.50d.txt" \
            --checkpoint_path="${CHECKPOINT_PATH}/conv_pmf_dataset_1/${LOG_DIR}/checkpoint/checkpoint_50.pt" \
            --batch_size=1024 \
            --window_size=5 \
            --n_word=128 \
            --n_factor=32 \
            --with_entropy=True \
            --entropy_threshold=${ENTROPY_THRESHOLD} \
            --least_act_num=50 \
            --k=30 \
            --use_cuda=True
    done
done
