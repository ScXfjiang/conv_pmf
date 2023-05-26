#!/bin/bash -l
#SBATCH --job-name=extract_words
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

DATA_PATH="/home/people/22200056/scratch/dataset/amazon/amazon_grocery_and_gourmet_foods_clean"
CHECKPOINT_PATH="/scratch/22200056/experiment/conv_pmf/original_embeds/n_factor_8/scripts/log/n_factor_8/0.0/May-11-2023-23-23-22-4b6cf107-19a8-4c27-9f92-7a83a6c6cf49/checkpoint/checkpoint_final.pt"

python ../src/extract_words.py \
    --dataset_path="${DATA_PATH}" \
    --word_embeds_path="/scratch/22200056/dataset/glove.6B/glove.6B.50d.txt" \
    --checkpoint_path="${CHECKPOINT_PATH}" \
    --n_factor=${n_factor} \
    --n_word=64 \
    --window_size=5 \
    --batch_size=1024 \
    --least_act_num=20 \
    --k=10 &
