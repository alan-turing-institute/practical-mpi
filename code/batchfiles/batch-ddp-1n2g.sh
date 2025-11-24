#!/bin/bash
#SBATCH --qos turing
#SBATCH --account vjgo8416-hpc2511
#SBATCH --time 0:10:0
#SBATCH --nodes 1
#SBATCH --gpus-per-node 2
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 16
#SBATCH --mem-per-gpu 16384
#SBATCH --job-name gpt2-ddp-1n2g
#SBATCH --output gpt2-ddp-1n2g-%j.out

# Execute using:
# sbatch batch-ddp-1n2g.sh

# Errors stop execution
set -e

module -q purge
module -q load baskerville
module -q load bask-apps/live
module -q load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

cd /bask/projects/v/vjgo8416-hpc2511/${USER}/hpc-training-nov-2025/3-Training/code

python3 -m venv --system-site-packages venv
source ./venv/bin/activate
pip -q install pip --upgrade
pip -q install -r requirements.txt

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
unset SLURM_MEM_PER_CPU
unset SLURM_TRES_PER_TASK

echo
echo "######################################"
echo "Starting:" ${SLURM_JOB_NAME}
echo "######################################"
echo

echo "SLURM_NNODES: ${SLURM_NNODES}"
echo "SLURM_GPUS_PER_NODE: ${SLURM_GPUS_PER_NODE}"

# Track GPU metrics
stdbuf -o0 nvidia-smi dmon -o TD -s puct -d 1 > gpu-${SLURM_JOB_ID}.txt &

# Execute the training
python -m torch.distributed.launch \
    --standalone \
    --nproc-per-node=${SLURM_GPUS_PER_NODE} \
    train_gpt2.py

echo
echo "######################################"
echo "Done"
echo "######################################"
echo

deactivate
