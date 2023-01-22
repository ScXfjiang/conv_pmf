#!/home/people/22200056/bin/zsh
#SBATCH --job-name=conv_pmf_test
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

# run from current directory
cd $SLURM_SUBMIT_DIR

DATA_PATH="/scratch/22200056/dataset/amazon/amazon_grocery_and_gourmet_foods"

CHECKPOINT_PATH="/home/people/22200056/scratch/experiment/conv_pmf_result/baseline_without_entropy/n_factor_32"
N_FACTOR=32
checkpoint_files=()
for ENTRY in "${CHECKPOINT_PATH}"/*; do
    checkpoint_files+="${ENTRY}/checkpoint/checkpoint_50.pt"
done
for checkpoint in ${checkpoint_files}; do
    python ../../src/test.py \
        --dataset_path="${DATA_PATH}" \
        --word_embeds_path="/scratch/22200056/dataset/glove.6B/glove.6B.50d.txt" \
        --global_user_id2global_user_idx="${DATA_PATH}/global_user_id2global_user_idx.pkl" \
        --global_item_id2global_item_idx="${DATA_PATH}/global_item_id2global_item_idx.pkl" \
        --checkpoint_path=${checkpoint} \
        --test_batch_size=256 \
        --window_size=5 \
        --n_word=128 \
        --n_factor=${N_FACTOR}
done
