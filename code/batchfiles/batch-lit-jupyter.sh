#!/bin/bash
#SBATCH --qos turing
#SBATCH --account vjgo8416-ml-workload
#SBATCH --time 0:10:0
#SBATCH --nodes 1
#SBATCH --gpus-per-node 1
#SBATCH --ntasks-per-node 1
#SBATCH --mem-per-cpu 16384
#SBATCH --job-name gpt2-lit-jupyter
#SBATCH --output gpt2-lit-jupyter-%j.out

# Execute using:
# sbatch batch-lit-jupyter.sh

module -q purge
module -q load baskerville
module -q load bask-apps/live
module -q load PyTorch/2.1.2-foss-2022b-CUDA-11.8.0

cd /bask/projects/v/vjgo8416-karpathy/${USER}/minGPT/gpt-2-video

python3 -m venv venv
source ./venv/bin/activate
pip -q install pip --upgrade
pip -q install -r requirements.txt

export OMP_NUM_THREADS=1

echo
echo "######################################"
echo "Starting:" ${SLURM_JOB_NAME}
echo "######################################"
echo

# Track GPU metrics
mpirun -host ${SLURM_JOB_NODELIST} bash -c 'nvidia-smi dmon -o TD -s puct -d 1 > gpu-${SLURM_JOB_ID}-${OMPI_COMM_WORLD_RANK}.txt' &

# Execute the training
srun python train_gpt2_lightning.py

echo
echo "######################################"
echo "Done"
echo "######################################"
echo

deactivate
