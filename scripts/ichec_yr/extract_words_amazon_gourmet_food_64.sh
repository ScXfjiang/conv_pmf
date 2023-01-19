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

DATA_PATH="/ichec/work/ucd01/yongru/dataset/amazon/amazon_grocery_and_gourmet_foods1"

checkpoint_files=()
checkpoint_files+="/ichec/work/ucd01/yongru/experiment/conv_pmf_result/baseline_without_entropy/n_factor_64/Oct-26-2022-23-53-23-2d2fc82c-af4c-4240-a194-23cce85109fd/checkpoint/checkpoint_50.pt"
checkpoint_files+=" /ichec/home/users/yongru/scratch/experiment/conv_pmf_result/conv_pmf_with_entropy/n_factor_64/1e-0/Oct-30-2022-05-23-12-0a9c85ff-39e7-4bb0-92b1-42bb3f40dcca/checkpoint/checkpoint_50.pt"
checkpoint_files+=" /ichec/home/users/yongru/scratch/experiment/conv_pmf_result/conv_pmf_with_entropy/n_factor_64/1e-1/Oct-30-2022-09-45-03-3a755cef-0b33-435b-a726-4d9b4ac3ded8/checkpoint/checkpoint_50.pt"
checkpoint_files+=" /ichec/home/users/yongru/scratch/experiment/conv_pmf_result/conv_pmf_with_entropy/n_factor_64/1e-2/Oct-30-2022-11-25-42-492c5740-8882-49b9-b34f-cb09054f6422/checkpoint/checkpoint_50.pt"
checkpoint_files+=" /ichec/home/users/yongru/scratch/experiment/conv_pmf_result/conv_pmf_with_entropy/n_factor_64/1e-3/Oct-26-2022-23-53-54-6f19eb2f-b85b-44d5-afd7-0cff561f2b74/checkpoint/checkpoint_50.pt"
checkpoint_files+=" /ichec/home/users/yongru/scratch/experiment/conv_pmf_result/conv_pmf_with_entropy/n_factor_64/1e-4/Oct-27-2022-03-12-59-c7da55bd-fd52-4201-b6d5-9483f85b2407/checkpoint/checkpoint_50.pt"
checkpoint_files+=" /ichec/home/users/yongru/scratch/experiment/conv_pmf_result/conv_pmf_with_entropy/n_factor_64/1e-5/Oct-27-2022-03-39-47-512ae4eb-9aa0-46df-905e-5eab297e665d/checkpoint/checkpoint_50.pt"
checkpoint_files+=" /ichec/home/users/yongru/scratch/experiment/conv_pmf_result/conv_pmf_with_entropy/n_factor_64/1e-6/Oct-27-2022-06-56-53-a6adf1ee-ab82-44f3-b689-c4aa315c3945/checkpoint/checkpoint_50.pt"
checkpoint_files+=" /ichec/home/users/yongru/scratch/experiment/conv_pmf_result/conv_pmf_with_entropy/n_factor_64/3e-0/Oct-31-2022-06-25-11-4f89c1d3-0c82-492c-b091-d6fa5ee118fe/checkpoint/checkpoint_50.pt"
checkpoint_files+=" /ichec/home/users/yongru/scratch/experiment/conv_pmf_result/conv_pmf_with_entropy/n_factor_64/5e-0/Oct-31-2022-06-35-53-2d0f8e1a-e45b-40a6-8ce7-1f82937ba5c2/checkpoint/checkpoint_50.pt"


for checkpoint in ${checkpoint_files}; do
    python ../../src/extract_words.py \
        --dataset_type="amazon_grocery_and_gourmet_foods" \
        --train_dataset_path="${DATA_PATH}/train.json" \
        --token_cnt_mat="${DATA_PATH}/token_cnt_mat.npz" \
        --word_embeds_path="/ichec/work/ucd01/yongru/dataset/glove.6B/glove.6B.50d.txt" \
        --checkpoint_path=${checkpoint} \
        --batch_size=1024 \
        --window_size=5 \
        --n_word=128 \
        --n_factor=64 \
        --with_entropy=False \
        --entropy_threshold=0 \
        --least_act_num=50 \
        --k=30
done
