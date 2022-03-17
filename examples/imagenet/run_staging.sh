VM_NAME=kmh-tpuvm-v3-128
# REPO=https://github.com/google/flax
REPO=https://71d519550fe3430ecbf39b70467e9210aed5da69:@github.com/KaimingHe/flax_dev.git
BRANCH=main
WORKDIR=gs://kmh-gcp/checkpoints/flax/examples/imagenet/$(date +%Y%m%d_%H%M)

# install venv
# pip3 install virtualenv
# python3 -m virtualenv env

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
# pip install 'jax[tpu]>=0.2.21' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
# pip install --upgrade clu
# python3 -c 'import jax; print(jax.device_count())'
# python3 -c 'import flax'
# cd ~/flax_dev &&
# git pull &&
cd ~/flax_dev/examples/imagenet &&
export TFDS_DATA_DIR=gs://kmh-gcp/tensorflow_datasets &&
python3 main.py \
    --workdir=$WORKDIR \
    --config=configs/tpu.py \
    --config.batch_size=32768
"

