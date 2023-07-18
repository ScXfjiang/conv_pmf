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
# specify the walltime e.g 192 hours
#SBATCH -t 192:00:00
# set to email at start,end and failed jobs
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sc.xfjiang@gmail.com

# run from current directory
cd $SLURM_SUBMIT_DIR

DATA_PATH="/home/people/22200056/scratch/dataset/amazon/amazon_grocery_and_gourmet_foods_clean"
CHECKPOINT_ROOT="/home/people/22200056/scratch/experiment/conv_pmf_results_final"

for n_factor in 6 8 10 12; do
    for epsilon in 0.0 0.4 0.8 1.2 1.6 2.0; do
        CHECKPOINT_DIR="${CHECKPOINT_ROOT}/n_factor_${n_factor}/${epsilon}"
        checkpoint_files=()
        for ENTRY in "${CHECKPOINT_DIR}"/*; do
            checkpoint_files+="${ENTRY}/checkpoint/checkpoint_final.pt"
        done
        for checkpoint in ${checkpoint_files}; do
            python ../src/extract_words.py \
                --dataset_path="${DATA_PATH}" \
                --word_embeds_path="/scratch/22200056/dataset/glove.6B/glove.6B.50d.txt" \
                --checkpoint_path="${checkpoint}" \
                --ref_token_cnt_mat="/scratch/22200056/dataset/wikitext/train_token_cnt_mat.npz" \
                --n_factor=${n_factor} \
                --n_word=64 \
                --window_size=5 \
                --strategy="all" \
                --batch_size=1024 \
                --least_act_num=20 \
                --k=10 \
                --log_dir_level_1="n_factor_${n_factor}" \
                --log_dir_level_2="${epsilon}"
        done
    done
done
