#!/bin/bash
#SBATCH --qos turing
#SBATCH --account vjgo8416-training25
#SBATCH --time 0:10:0
#SBATCH --nodes 1
#SBATCH --gpus-per-node 2
#SBATCH --cpus-per-gpu 9
#SBATCH --mem 16384
#SBATCH --job-name gpt2-std-1n2g
#SBATCH --output gpt2-std-1n2g-%j.out

# Execute using:
# sbatch batch-std-1n2g.sh

module -p purge
module -p load baskerville
module -p load bask-apps/live
module -p load PyTorch/2.1.2-foss-2022b-CUDA-11.8.0

cd /bask/projects/v/vjgo8416-training25/${USER}/practical-mpi/code

python3 -m venv venv
source ./venv/bin/activate
pip -p install pip --upgrade
pip -p install -r requirements.txt

export OMP_NUM_THREADS=${SLURM_CPUS_PER_GPU}

echo
echo "######################################"
echo "Starting:" ${SLURM_JOB_NAME}
echo "######################################"
echo

# Track GPU metrics
stdbuf -o0 nvidia-smi dmon -o TD -s puct -d 1 > gpu-${SLURM_JOB_ID}.txt &

# Execute the training
python train_gpt2.py

echo
echo "######################################"
echo "Done"
echo "######################################"
echo

deactivate
