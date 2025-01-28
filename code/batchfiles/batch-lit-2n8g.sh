#!/bin/bash
#SBATCH --qos turing
#SBATCH --account vjgo8416-training25
#SBATCH --time 0:10:0
#SBATCH --nodes 2
#SBATCH --gpus-per-node 4
#SBATCH --ntasks-per-node 4
#SBATCH --cpus-per-gpu 9
#SBATCH --mem 16384
#SBATCH --job-name gpt2-lit-2n8g
#SBATCH --output gpt2-lit-2n8g-%j.out

# Execute using:
# sbatch batch-lit-2n8g.sh

module -q purge
module -q load baskerville
module -q load bask-apps/live
module -q load PyTorch/2.1.2-foss-2022b-CUDA-11.8.0

cd /bask/projects/v/vjgo8416-training25/${USER}/practical-mpi/code

python3 -m venv venv
source ./venv/bin/activate
pip -q install pip --upgrade
pip -q install -r requirements.txt

export OMP_NUM_THREADS=${SLURM_CPUS_PER_GPU}

echo
echo "######################################"
echo "Starting:" ${SLURM_JOB_NAME}
echo "######################################"
echo

# Track GPU metrics
mpirun -host ${SLURM_JOB_NODELIST} bash -c 'stdbuf -o0 nvidia-smi dmon -o TD -s puct -d 1 > gpu-${SLURM_JOB_ID}-${OMPI_COMM_WORLD_RANK}.txt' &

# Execute the training
srun python train_gpt2.py

echo
echo "######################################"
echo "Done"
echo "######################################"
echo

deactivate
