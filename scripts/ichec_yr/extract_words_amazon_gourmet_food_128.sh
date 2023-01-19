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
checkpoint_files+="/ichec/home/users/yongru/scratch/experiment/conv_pmf_result/baseline_without_entropy/n_factor_128/Oct-26-2022-23-57-32-e860c154-8162-47b5-b7de-458d1c5ce6b1/checkpoint/checkpoint_50.pt"
checkpoint_files+=" /ichec/home/users/yongru/scratch/experiment/conv_pmf_result/conv_pmf_with_entropy/n_factor_128/1e-0/Oct-30-2022-11-27-49-91629287-0300-4078-8dbc-dfdf3e5006d5/checkpoint/checkpoint_50.pt"
checkpoint_files+=" /ichec/home/users/yongru/scratch/experiment/conv_pmf_result/conv_pmf_with_entropy/n_factor_128/1e-1/Oct-30-2022-14-07-19-0e169413-767d-47f9-b85e-7b69147f2fa4/checkpoint/checkpoint_50.pt"
checkpoint_files+=" /ichec/home/users/yongru/scratch/experiment/conv_pmf_result/conv_pmf_with_entropy/n_factor_128/1e-2/Oct-30-2022-17-48-01-a487105b-04fa-43ec-a1b8-9e99fa6771c6/checkpoint/checkpoint_50.pt"
checkpoint_files+=" /ichec/home/users/yongru/scratch/experiment/conv_pmf_result/conv_pmf_with_entropy/n_factor_128/1e-3/Oct-26-2022-23-57-31-1dabddae-72d4-45a5-90e7-7ee1ce2e610a/checkpoint/checkpoint_50.pt"
checkpoint_files+=" /ichec/home/users/yongru/scratch/experiment/conv_pmf_result/conv_pmf_with_entropy/n_factor_128/1e-4/Oct-27-2022-03-25-49-72d364ea-f179-4515-9562-ba68bf105548/checkpoint/checkpoint_50.pt"
checkpoint_files+=" /ichec/home/users/yongru/scratch/experiment/conv_pmf_result/conv_pmf_with_entropy/n_factor_128/1e-5/Oct-27-2022-03-59-06-6f55ff40-3ce0-42d9-90b2-db49c17a2f64/checkpoint/checkpoint_50.pt"
checkpoint_files+=" /ichec/home/users/yongru/scratch/experiment/conv_pmf_result/conv_pmf_with_entropy/n_factor_128/1e-6/Oct-27-2022-07-27-02-85275717-9f14-44bf-8ee4-c64fd9a3bc43/checkpoint/checkpoint_50.pt"
checkpoint_files+=" /ichec/home/users/yongru/scratch/experiment/conv_pmf_result/conv_pmf_with_entropy/n_factor_128/3e-0/Oct-31-2022-11-19-41-102ef829-afdc-4627-8f0f-17dc51f59711/checkpoint/checkpoint_50.pt"
checkpoint_files+=" /ichec/home/users/yongru/scratch/experiment/conv_pmf_result/conv_pmf_with_entropy/n_factor_128/5e-0/Oct-31-2022-12-48-45-996cd229-dc81-4e72-bdb1-f4c60c7f0b69/checkpoint/checkpoint_50.pt"

for checkpoint in ${checkpoint_files}; do
    python ../../src/extract_words.py \
        --dataset_type="amazon_grocery_and_gourmet_foods" \
        --train_dataset_path="${DATA_PATH}/train.json" \
        --token_cnt_mat="${DATA_PATH}/token_cnt_mat.npz" \
        --word_embeds_type="glove.6B.50d" \
        --word_embeds_path="/ichec/work/ucd01/yongru/dataset/glove.6B/glove.6B.50d.txt" \
        --checkpoint_path=${checkpoint} \
        --batch_size=1024 \
        --window_size=5 \
        --n_word=128 \
        --n_factor=128 \
        --with_entropy=False \
        --entropy_threshold=0 \
        --least_act_num=50 \
        --k=30
done
