VM_NAME=kmh-tpuvm-v3-128
# REPO=https://github.com/google/flax
REPO=https://71d519550fe3430ecbf39b70467e9210aed5da69:@github.com/KaimingHe/flax_dev.git
BRANCH=main

CONFIG=tpu_vit_base
WORKDIR=gs://kmh-gcp/checkpoints/flax/examples/imagenet/$CONFIG_$(date +%Y%m%d_%H%M)

## install conda
# apt-get install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6

## install jax and flax
# pip install 'jax[tpu]>=0.2.21' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html &&
# pip install --upgrade clu &&
# python3 -c 'import jax; print(jax.device_count())' &&
# python3 -c 'import flax' &&

## clone the repo
# git config --global credential.helper store &&
# git clone -b $BRANCH $REPO &&

gcloud alpha compute tpus tpu-vm ssh ${VM_NAME} --zone europe-west4-a \
    --worker=all --command "
# git config --global credential.helper store &&
# git clone -b $BRANCH $REPO &&

# pip install 'jax[tpu]>=0.2.21' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
# pip install --upgrade clu

echo start
# cd ~/flax_dev
# git checkout vit
# git pull

# cd ~/flax_dev/examples/imagenet
sudo pkill python
# source run_kill.sh
# sudo lsof -w /dev/accel0

# python3 -c 'import jax; print(jax.device_count())'
# python3 -c 'import flax'

# cd ~/flax_dev/examples/imagenet
# export TFDS_DATA_DIR=gs://kmh-gcp/tensorflow_datasets
# python3 main.py \
#     --workdir=$WORKDIR \
#     --config=configs/$CONFIG.py \
#     --config.batch_size=32768
"

