#!/bin/bash -l
#SBATCH --job-name=extract_words
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

DATA_PATH="/ichec/home/users/xfjiang/scratch/dataset/amazon/amazon_grocery_and_gourmet_foods1"

checkpoint_files=()
checkpoint_files+="/ichec/work/ucd01/xfjiang/experiment/conv_pmf_result/baseline_without_entropy/n_factor_32/Oct-26-2022-23-48-17-27b6bf4c-e922-4a24-8c52-9d77e1567651/checkpoint/checkpoint_50.pt"
checkpoint_files+=" /ichec/work/ucd01/xfjiang/experiment/conv_pmf_result/conv_pmf_with_entropy/n_factor_32/1e-0/Oct-30-2022-01-07-49-0a907a67-9d95-48bd-a71e-e20e9cb854c6/checkpoint/checkpoint_50.pt"
checkpoint_files+=" /ichec/work/ucd01/xfjiang/experiment/conv_pmf_result/conv_pmf_with_entropy/n_factor_32/1e-1/Oct-30-2022-05-11-17-0b626569-0947-454d-8448-02327c4ed08c/checkpoint/checkpoint_50.pt"
checkpoint_files+=" /ichec/work/ucd01/xfjiang/experiment/conv_pmf_result/conv_pmf_with_entropy/n_factor_32/1e-2/Oct-30-2022-05-13-26-0918b868-6033-4d04-8ab9-13e4440f3ffe/checkpoint/checkpoint_50.pt"
checkpoint_files+=" /ichec/work/ucd01/xfjiang/experiment/conv_pmf_result/conv_pmf_with_entropy/n_factor_32/1e-3/Oct-27-2022-05-44-32-599c72c7-59d8-4388-a0d7-085b6b62edc3/checkpoint/checkpoint_50.pt"
checkpoint_files+=" /ichec/work/ucd01/xfjiang/experiment/conv_pmf_result/conv_pmf_with_entropy/n_factor_32/1e-4/Oct-27-2022-08-43-08-a58053f7-fe57-4896-a09e-8da287fc75d2/checkpoint/checkpoint_50.pt"
checkpoint_files+=" /ichec/work/ucd01/xfjiang/experiment/conv_pmf_result/conv_pmf_with_entropy/n_factor_32/1e-5/Oct-27-2022-12-01-31-fc30ef33-3701-4b03-adf8-dcba9689470a/checkpoint/checkpoint_50.pt"
checkpoint_files+=" /ichec/work/ucd01/xfjiang/experiment/conv_pmf_result/conv_pmf_with_entropy/n_factor_32/1e-6/Oct-27-2022-12-55-24-7cbc2c96-2f03-4c68-b4dd-bba06f36a506/checkpoint/checkpoint_50.pt"
checkpoint_files+=" /ichec/work/ucd01/xfjiang/experiment/conv_pmf_result/conv_pmf_with_entropy/n_factor_32/3e-0/Oct-31-2022-00-21-28-2f08f504-a628-4f23-8e45-63c3ac10a015/checkpoint/checkpoint_50.pt"
checkpoint_files+=" /ichec/work/ucd01/xfjiang/experiment/conv_pmf_result/conv_pmf_with_entropy/n_factor_32/5e-0/Oct-31-2022-05-02-24-e1eb179a-5ce4-48f0-b1ac-eae817284754/checkpoint/checkpoint_50.pt"

for checkpoint in ${checkpoint_files}; do
    python ../../src/extract_words.py \
        --dataset_type="amazon_grocery_and_gourmet_foods" \
        --train_dataset_path="${DATA_PATH}/train.json" \
        --token_cnt_mat="${DATA_PATH}/token_cnt_mat.npz" \
        --word_embeds_type="glove.6B.50d" \
        --word_embeds_path="/ichec/work/ucd01/xfjiang/dataset/glove.6B/glove.6B.50d.txt" \
        --checkpoint_path=${checkpoint} \
        --batch_size=1024 \
        --window_size=5 \
        --n_word=128 \
        --n_factor=32 \
        --with_entropy=False \
        --entropy_threshold=0 \
        --least_act_num=50 \
        --k=30
done
