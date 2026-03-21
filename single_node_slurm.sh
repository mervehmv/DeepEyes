#!/bin/bash
#SBATCH --job-name=eval_grounded
#SBATCH --mem=350G
#SBATCH --partition=kolyoz-cuda
#SBATCH --nodelist=kolyoz29
#SBATCH --time=24:00:00
#SBATCH -A proj46
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:4
#SBATCH -c 64
#SBATCH -C H200
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

verl_workdir="/arf/scratch/hvural/DeepEyes"
home_dir="/arf/scratch/hvural"
apptainer_image_path="/arf/scratch/hvural/verl_vllm017_latest.sif"
PROJECT_NAME="agent_vlagent"
EXPERIMENT_NAME="qwen25_vl_7b_instruct_vl_agent_v2"
export LLM_AS_A_JUDGE_BASE="http://kolyoz25.ib:8008/v1"
# export TOKENIZERS_PARALLELISM=false
export SAVE_CHECKPOINT_DIR=/arf/scratch/hvural/DeepEyes/checkpoints
export WORLD_SIZE=1
# NOTE: refer to https://verl.readthedocs.io/en/latest/README_vllm0.8.html
# export VLLM_ATTENTION_BACKEND=XFORMERS # vllm + qwen2-7b with flash_attn has some issues

DATA_DIR=/arf/scratch/hvural/DeepEyes/caption_dataset_rl_v6_env_clean
DEBUG_DATASET_TEST=/arf/scratch/hvural/DeepEyes/caption_dataset_rl_v6_env_clean/test-00000-of-00002.parquet

VISUAL_DATASET_TRAIN_1=/arf/scratch/hvural/DeepEyes/caption_dataset_rl_v6_env_clean/train-00000-of-00008.parquet
VISUAL_DATASET_TRAIN_2=/arf/scratch/hvural/DeepEyes/caption_dataset_rl_v6_env_clean/train-00001-of-00008.parquet
VISUAL_DATASET_TRAIN_3=/arf/scratch/hvural/DeepEyes/caption_dataset_rl_v6_env_clean/train-00002-of-00008.parquet
VISUAL_DATASET_TRAIN_4=/arf/scratch/hvural/DeepEyes/caption_dataset_rl_v6_env_clean/train-00003-of-00008.parquet
VISUAL_DATASET_TRAIN_5=/arf/scratch/hvural/DeepEyes/caption_dataset_rl_v6_env_clean/train-00004-of-00008.parquet
VISUAL_DATASET_TRAIN_6=/arf/scratch/hvural/DeepEyes/caption_dataset_rl_v6_env_clean/train-00005-of-00008.parquet
VISUAL_DATASET_TRAIN_7=/arf/scratch/hvural/DeepEyes/caption_dataset_rl_v6_env_clean/train-00006-of-00008.parquet
VISUAL_DATASET_TRAIN_8=/arf/scratch/hvural/DeepEyes/caption_dataset_rl_v6_env_clean/train-00007-of-00008.parquet

VISUAL_DATASET_VAL_1=/arf/scratch/hvural/DeepEyes/caption_dataset_rl_v6_env_clean/test-00000-of-00002.parquet
VISUAL_DATASET_VAL_2=/arf/scratch/hvural/DeepEyes/caption_dataset_rl_v6_env_clean/test-00001-of-00002.parquet

REF_MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct
# GPUs: prefer CUDA_VISIBLE_DEVICES because it's what Slurm actually assigned
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  NUM_GPUS="$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')"
elif [[ -n "${SLURM_JOB_GPUS:-}" ]]; then
  NUM_GPUS="$(echo "${SLURM_JOB_GPUS}" | awk -F',' '{print NF}')"
elif [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
  NUM_GPUS="${SLURM_GPUS_ON_NODE}"
else
  NUM_GPUS="0"
fi

NUM_CPUS="${SLURM_CPUS_PER_TASK:-64}"

# Robust IP detection with fallback
head_node_ip="$(ip route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="src") print $(i+1)}' | head -n1)"
if [[ -z "${head_node_ip}" ]]; then
  head_node_ip="$(hostname -i | awk '{print $1}')"
fi

# Port selection: Try 6379 first, fallback to random if busy
port=$(python3 -c "import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.bind(('', 6379))
    print(6379)
except OSError:
    sock.bind(('', 0))  # Bind to any free port
    print(sock.getsockname()[1])
finally:
    sock.close()")

echo "Detected NUM_GPUS=${NUM_GPUS} NUM_CPUS=${NUM_CPUS}"
echo "Head IP=${head_node_ip}:${port}"
echo "CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-}' SLURM_JOB_GPUS='${SLURM_JOB_GPUS:-}' SLURM_GPUS_ON_NODE='${SLURM_GPUS_ON_NODE:-}'"

srun --nodes=1 --ntasks=1 apptainer exec --nv \
  --bind "${verl_workdir}":"${verl_workdir}" \
  --bind "${home_dir}":"${home_dir}" \
  "${apptainer_image_path}" \
  bash -lc "
    set -euo pipefail
    
    # Use a unique temp directory to avoid permission collisions
    export RAY_TMPDIR=/tmp/ray_${SLURM_JOB_ID:-local}
    mkdir -p \"\$RAY_TMPDIR\"

    cleanup() { 
        echo \"Stopping Ray...\"
        ray stop --force || true
        # Optional: rm -rf \"\$RAY_TMPDIR\" 
    }
    trap cleanup EXIT SIGINT SIGTERM

    ray stop --force || true
    ray start --head --node-ip-address='${head_node_ip}' --port=${port} \
      --num-cpus=${NUM_CPUS} --num-gpus=${NUM_GPUS} --temp-dir \"\$RAY_TMPDIR\" --disable-usage-stats

    set -x

PROJECT_NAME="agent_vlagent"
EXPERIMENT_NAME="qwen25_vl_7b_instruct_vl_agent_v2"

export SAVE_CHECKPOINT_DIR=/arf/scratch/hvural/DeepEyes/checkpoints

REF_MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct

python3 -m verl.trainer.main_ppo \
    +debug=False \
    +vs_debug=False \
    "data.train_files=[\"${VISUAL_DATASET_TRAIN_1}\",\"${VISUAL_DATASET_TRAIN_2}\",\"${VISUAL_DATASET_TRAIN_3}\",\"${VISUAL_DATASET_TRAIN_4}\",\"${VISUAL_DATASET_TRAIN_5}\",\"${VISUAL_DATASET_TRAIN_6}\",\"${VISUAL_DATASET_TRAIN_7}\",\"${VISUAL_DATASET_TRAIN_8}\"]" \
    "data.val_files=[\"${VISUAL_DATASET_VAL_1}\",\"${VISUAL_DATASET_VAL_2}\"]" \
    data.train_batch_size=128 \
    data.max_prompt_length=10240 \
    data.max_response_length=10240 \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.model.path=${REF_MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0001 \
    "actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra]" \
    "actor_rollout_ref.actor.checkpoint.load_contents=[model,optimizer,extra]" \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    agent.activate_agent=True \
    agent.tool_name_key=env_name \
    agent.single_response_max_tokens=2048 \
    agent.max_turns=9 \
    agent.concurrent_workers=1 \
    agent.show_tqdm=True \
    trainer.critic_warmup=0 \
    "trainer.logger=[console,wandb]" \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.nnodes=${WORLD_SIZE} \
    trainer.save_freq=50 \
    trainer.test_freq=-1 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir=${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME} \
    trainer.total_epochs=5
  " 2>&1 | tee eval_grounded.log
