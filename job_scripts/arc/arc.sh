### ADDITIONAL RUN INFO ###
#SBATCH --array=0
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
### Support for requeueing on preemption ###
### LOG INFO ###
#SBATCH --job-name=arc_v1
#SBATCH --output=logs/slurm/arc/%x_%A-%a.log
export RUN_NAME="${SLURM_JOB_NAME}"
# NOTE ctrl d ALL THREE of above to modify job-name, output, and RUN_NAME (which should all be the same)
export MODEL_NAME="${RUN_NAME%%-*}"
export MODEL_SIZE="${RUN_NAME#*-}"; export MODEL_SIZE="${MODEL_SIZE%%-*}"
mkdir -p logs/slurm/arc/
module purge

module load python/3.12.11-fasrc01
source ~/.bashrc
mamba activate hrm
lr=(0.00001 0.00005)
alpha=(10 10)
# remember to set rms norm if desired
# alpha_lr=(1500)

python train_model.py \
	--run_name ${RUN_NAME}_lr_${lr[${SLURM_ARRAY_TASK_ID}]}_steps_${alpha[${SLURM_ARRAY_TASK_ID}]}_bs_32_RMSNorm \
	--modality "ARC" \
	--model_name "ebm" \
	--normalize_initial_condition \
	--denoising_initial_condition "random_noise" \
	--mcmc_step_size ${alpha[${SLURM_ARRAY_TASK_ID}]} \
	--mcmc_num_steps 10 \
	--grid_height 30 \
	--grid_width 30 \
	--grid_channels 12 \
	--grid_hidden_dim 384 \
	--grid_num_res_layers 3 \
	--grid_num_indices 400 \
	--grid_index_embed_dim 64 \
	--grid_mlp_hidden_dim 128 \
	--gpus "-1" \
	--check_val_every_n_epoch 10 \
	--peak_learning_rate ${lr[${SLURM_ARRAY_TASK_ID}]} \
	--batch_size_per_device 8 \
	--accumulate_grad_batches 4 \
	--gradient_clip_val 1.0 \
	--weight_decay 0 \
	--max_steps 1000000 \
	--dataset_name "arc" \
	--num_workers 4 \
	--wandb_project 'arc_pretrain' \
	--log_model_archi \
	--log_gradients \
	--wandb_watch \
	--use_rmsnorm
	${SLURM_ARRAY_TASK_ID:+--is_slurm_run}