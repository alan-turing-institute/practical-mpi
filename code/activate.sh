# Check that the fils is being sourced
(return 0 2>/dev/null) && sourced=1 || sourced=0
if [[ sourced -eq 0 ]]; then
  echo "Please source the file, like this: "
  echo "source ./activate.sh"
  exit 0
fi

echo "Loading modules"
module -q purge
module -q load baskerville
module -q load bask-apps/live
module -q load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

echo "Activating venv"
python3 -m venv --system-site-packages venv
source ./venv/bin/activate
pip -q install pip --upgrade
pip -q install -r requirements.txt
