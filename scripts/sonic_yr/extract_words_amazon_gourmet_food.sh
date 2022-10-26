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
# specify the walltime e.g 96 hours
#SBATCH -t 96:00:00
# set to email at start,end and failed jobs

# run from current directory
cd $SLURM_SUBMIT_DIR

DATA_PATH="/home/people/22200056/workspace/dataset/amazon/amazon_grocery_and_gourmet_foods1"

python ../../src/extract_words.py \
    --dataset_type="amazon_grocery_and_gourmet_foods" \
    --train_dataset_path="${DATA_PATH}/train.json" \
    --word_embeds_type="glove.6B.50d" \
    --word_embeds_path="/home/people/22200056/workspace/dataset/glove.6B/glove.6B.50d.txt" \
    --checkpoint_path="/scratch/22200056/conv_pmf_results/conv_pmf_entropy_A/epsilon_0.00005/Sep-11-2022-05-19-09/checkpoint/checkpoint_50.pt" \
    --batch_size=1024 \
    --window_size=5 \
    --n_word=128 \
    --n_factor=32 \
    --with_entropy=False \
    --entropy_threshold=0.0 \
    --least_act_num=50 \
    --k=30 \
    --use_cuda=True