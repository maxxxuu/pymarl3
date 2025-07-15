#!/bin/bash
#SBATCH --job-name=SMAC_repeat
#SBATCH -C a100
#SBATCH --nodes=1                    # we request one node
#SBATCH --ntasks-per-node=1          # with one task per node (= number of GPUs here)
#SBATCH --time=20:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16           # number of cores per task for gpu_p5 (1/8 of 8-GPUs A100 node)
#SBATCH --hint=nomultithread         # hyperthreading is deactivated
#SBATCH --array=0-39
##SBATCH --mail-type=FAIL
##SBATCH --mail-user=

# Cleans out the modules loaded in interactive and inherited by default
module purge

module load arch/a100
module load python/3.9.18
module load cuda/12.1

cd $WORK/pymarl3

# 8 setting × 5 repeat
combo_id=$((SLURM_ARRAY_TASK_ID / 5))
read algo map <<< $(sed -n "$((combo_id + 1))p" configs_list.txt)

python -u src/main.py \
    --config=$algo \
    --env-config=sc2 with \
    env_args.map_name=$map \
    obs_agent_id=True \
    obs_last_action=False \
    runner=parallel \
    batch_size_run=8 \
    buffer_size=5000 \
    t_max=10050000 \
    epsilon_anneal_time=100000 \
    batch_size=128 \
    td_lambda=0.6