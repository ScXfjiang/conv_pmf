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
checkpoint_files+="/ichec/work/ucd01/xfjiang/experiment/conv_pmf_result/baseline_without_entropy/n_factor_8/Oct-26-2022-23-45-04-750cfdf6-424b-42bb-b08a-1965bee1b2f6/checkpoint/checkpoint_50.pt"
checkpoint_files+=" /ichec/work/ucd01/xfjiang/experiment/conv_pmf_result/conv_pmf_with_entropy/n_factor_8/1e-0/Oct-29-2022-17-49-37-7e8ce4f8-953e-4a17-a9c4-ef14de1e4114/checkpoint/checkpoint_50.pt"
checkpoint_files+=" /ichec/work/ucd01/xfjiang/experiment/conv_pmf_result/conv_pmf_with_entropy/n_factor_8/1e-1/Oct-29-2022-17-49-43-abfbd64f-2b0c-4cb7-8b57-21e8b5c66f03/checkpoint/checkpoint_50.pt"
checkpoint_files+=" /ichec/work/ucd01/xfjiang/experiment/conv_pmf_result/conv_pmf_with_entropy/n_factor_8/1e-2/Oct-29-2022-17-49-34-9f4dd269-20f5-4ef0-899d-2f75231717d4/checkpoint/checkpoint_50.pt"
checkpoint_files+=" /ichec/work/ucd01/xfjiang/experiment/conv_pmf_result/conv_pmf_with_entropy/n_factor_8/1e-3/Oct-26-2022-23-47-02-cd7aa997-36a3-4920-860d-67ad4ede19fc/checkpoint/checkpoint_50.pt"
checkpoint_files+=" /ichec/work/ucd01/xfjiang/experiment/conv_pmf_result/conv_pmf_with_entropy/n_factor_8/1e-4/Oct-27-2022-03-48-55-f721b163-aa41-4013-81ad-e64f39e81cb1/checkpoint/checkpoint_50.pt"
checkpoint_files+=" /ichec/work/ucd01/xfjiang/experiment/conv_pmf_result/conv_pmf_with_entropy/n_factor_8/1e-5/Oct-27-2022-04-09-16-0b18fadd-680b-4f03-8b6f-5db0eab70fc6/checkpoint/checkpoint_50.pt"
checkpoint_files+=" /ichec/work/ucd01/xfjiang/experiment/conv_pmf_result/conv_pmf_with_entropy/n_factor_8/1e-6/Oct-27-2022-05-35-22-d8fb1c3d-4e05-4ccf-8ae8-5a5e66672a01/checkpoint/checkpoint_50.pt"
checkpoint_files+=" /ichec/work/ucd01/xfjiang/experiment/conv_pmf_result/conv_pmf_with_entropy/n_factor_8/3e-0/Oct-30-2022-18-03-56-b27285df-02b5-4977-9a26-e2ce169f920f/checkpoint/checkpoint_50.pt"
checkpoint_files+=" /ichec/work/ucd01/xfjiang/experiment/conv_pmf_result/conv_pmf_with_entropy/n_factor_8/5e-0/Oct-30-2022-18-41-39-02a38bc1-7d50-4898-b459-8285e4b34f88/checkpoint/checkpoint_50.pt"

for checkpoint in ${checkpoint_files}; do
    python ../../src/extract_words.py \
        --dataset_path="${DATA_PATH}" \
        --token_cnt_mat="${DATA_PATH}/token_cnt_mat.npz" \
        --word_embeds_path="/ichec/work/ucd01/xfjiang/dataset/glove.6B/glove.6B.50d.txt" \
        --checkpoint_path=${checkpoint} \
        --batch_size=1024 \
        --window_size=5 \
        --n_word=128 \
        --n_factor=8 \
        --with_entropy=False \
        --entropy_threshold=0 \
        --least_act_num=50 \
        --k=30
done
