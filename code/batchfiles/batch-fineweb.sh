#!/bin/bash
#SBATCH --qos turing
#SBATCH --account vjgo8416-karpathy
#SBATCH --time 3:00:0
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --cpus-per-gpu 36
#SBATCH --mem 16384
#SBATCH --job-name fineweb-download
#SBATCH --output fineweb-%j.out

# Execute using:
# sbatch batch-fineweb.sh

module -p purge
module -p load baskerville
module -p load bask-apps/live
module -p load PyTorch/2.1.2-foss-2022b-CUDA-11.8.0

cd /bask/projects/v/vjgo8416-karpathy/${USER}/minGPT/gpt-2-video

python3 -m venv venv
source ./venv/bin/activate
pip -p install pip --upgrade
pip -p install -r requirements.txt

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
