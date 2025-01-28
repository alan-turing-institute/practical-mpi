#!/bin/bash
#SBATCH --qos turing
#SBATCH --account vjgo8416-karpathy
#SBATCH --time 0:10:0
#SBATCH --nodes 2
#SBATCH --gpus-per-node 1
#SBATCH --cpus-per-gpu 9
#SBATCH --mem 16384
#SBATCH --job-name gpt2-ddp-2n2g
#SBATCH --output gpt2-ddp-2n2g-%j.out

# Execute using:
# sbatch batch-ddp-1n2g.sh

module -q purge
module -q load baskerville
module -q load bask-apps/live
module -q load PyTorch/2.1.2-foss-2022b-CUDA-11.8.0

cd /bask/projects/v/vjgo8416-karpathy/${USER}/minGPT/gpt-2-video

python3 -m venv venv
source ./venv/bin/activate
pip -q install pip --upgrade
pip -q install -r requirements.txt

export OMP_NUM_THREADS=${SLURM_CPUS_PER_GPU}
export MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MASTER_PORT=$((1024 + $RANDOM % 64511))

echo
echo "######################################"
echo "Starting:" ${SLURM_JOB_NAME}
echo "######################################"
echo

# Track GPU metrics
mpirun -host ${SLURM_JOB_NODELIST} bash -c 'stdbuf -o0 nvidia-smi dmon -o TD -s puct -d 1 > gpu-${SLURM_JOB_ID}-${OMPI_COMM_WORLD_RANK}.txt' &

# Execute the training
python -m torch.distributed.launch --nproc_per_node=${SLURM_GPUS_PER_NODE} --nnodes=${SLURM_NNODES} --master-port=${MASTER_PORT} --master-addr=${MASTER_ADDR} train_gpt2_ddp.py

echo
echo "######################################"
echo "Done"
echo "######################################"
echo

deactivate
