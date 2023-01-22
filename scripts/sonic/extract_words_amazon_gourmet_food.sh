#!/home/people/22200056/bin/zsh
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

DATA_PATH="/home/people/22200056/scratch/dataset/amazon/amazon_grocery_and_gourmet_foods"

python ../../src/extract_words.py \
    --dataset_path="${DATA_PATH}" \
    --token_cnt_mat="${DATA_PATH}/token_cnt_mat.npz" \
    --word_embeds_path="/scratch/22200056/dataset/glove.6B/glove.6B.50d.txt" \
    --checkpoint_path="/scratch/22200056/experiment/compare/n_factor_32_epsilon_0/scripts/sonic/log/Jan-20-2023-17-36-34-add12c04-6daf-4063-941e-6514ff6a0f2e/checkpoint/checkpoint_50.pt" \
    --batch_size=1024 \
    --window_size=5 \
    --n_word=128 \
    --n_factor=32 \
    --least_act_num=50 \
    --k=30
