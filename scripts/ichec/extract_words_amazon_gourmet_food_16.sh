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
checkpoint_files+="/ichec/work/ucd01/xfjiang/experiment/conv_pmf_result/baseline_without_entropy/n_factor_16/Oct-26-2022-23-47-02-95b9d8c4-086e-48e7-8574-370c6f0c001e/checkpoint/checkpoint_50.pt"
checkpoint_files+=" /ichec/work/ucd01/xfjiang/experiment/conv_pmf_result/conv_pmf_with_entropy/n_factor_16/1e-0/Oct-29-2022-21-56-04-0ee8928b-e8d0-4bf8-901e-3584ed400517/checkpoint/checkpoint_50.pt"
checkpoint_files+=" /ichec/work/ucd01/xfjiang/experiment/conv_pmf_result/conv_pmf_with_entropy/n_factor_16/1e-1/Oct-29-2022-23-59-35-1cd28ed2-38c4-4d50-89c4-f265f2854c84/checkpoint/checkpoint_50.pt"
checkpoint_files+=" /ichec/work/ucd01/xfjiang/experiment/conv_pmf_result/conv_pmf_with_entropy/n_factor_16/1e-2/Oct-29-2022-23-59-53-291edf9c-82f9-4f15-8605-f78d5b04a25f/checkpoint/checkpoint_50.pt"
checkpoint_files+=" /ichec/work/ucd01/xfjiang/experiment/conv_pmf_result/conv_pmf_with_entropy/n_factor_16/1e-3/Oct-27-2022-08-11-47-93143d36-9010-454a-9d10-84c1cad73230/checkpoint/checkpoint_50.pt"
checkpoint_files+=" /ichec/work/ucd01/xfjiang/experiment/conv_pmf_result/conv_pmf_with_entropy/n_factor_16/1e-4/Oct-27-2022-08-32-27-6826fb2b-5681-4ed8-9e80-281aca0ebce2/checkpoint/checkpoint_50.pt"
checkpoint_files+=" /ichec/work/ucd01/xfjiang/experiment/conv_pmf_result/conv_pmf_with_entropy/n_factor_16/1e-5/Oct-27-2022-11-45-48-f06449e3-9739-4df1-a735-4ac9a049916f/checkpoint/checkpoint_50.pt"
checkpoint_files+=" /ichec/work/ucd01/xfjiang/experiment/conv_pmf_result/conv_pmf_with_entropy/n_factor_16/1e-6/Oct-27-2022-12-33-00-3fbdcefe-c85d-4dcf-8184-3623edb7e7af/checkpoint/checkpoint_50.pt"
checkpoint_files+=" /ichec/work/ucd01/xfjiang/experiment/conv_pmf_result/conv_pmf_with_entropy/n_factor_16/3e-0/Oct-30-2022-22-49-36-6402a407-16c8-41f5-b2a8-76785d62932b/checkpoint/checkpoint_50.pt"
checkpoint_files+=" /ichec/work/ucd01/xfjiang/experiment/conv_pmf_result/conv_pmf_with_entropy/n_factor_16/5e-0/Oct-31-2022-00-13-20-7c8013e1-95fa-44f3-9112-fab3abeb5d3d/checkpoint/checkpoint_50.pt"

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
        --n_factor=16 \
        --with_entropy=False \
        --entropy_threshold=0 \
        --least_act_num=50 \
        --k=30 \
        --use_cuda=True
done
