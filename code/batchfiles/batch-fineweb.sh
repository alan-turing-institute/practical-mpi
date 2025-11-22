#!/bin/bash
#SBATCH --qos turing
#SBATCH --account vjgo8416-hpc2511
#SBATCH --time 3:00:0
#SBATCH --nodes 2
#SBATCH --gpus-per-node 1
#SBATCH --ntasks-per-gpu 1
#SBATCH --cpus-per-task 9
#SBATCH --mem-per-gpu 16384
#SBATCH --job-name fineweb-download
#SBATCH --output fineweb-%j.out

# Execute using:
# sbatch batch-fineweb.sh

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

echo
echo "######################################"
echo "Starting"
echo "######################################"
echo

# Execute the download and sharding process
python3 fineweb.py

echo
echo "######################################"
echo "Done"
echo "######################################"
echo

deactivate
