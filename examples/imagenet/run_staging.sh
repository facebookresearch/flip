# VM_NAME=kmh-tpuvm-v3-128
VM_NAME=kmh-tpuvm-v3-128
echo $VM_NAME
# REPO=https://github.com/google/flax
REPO=https://71d519550fe3430ecbf39b70467e9210aed5da69:@github.com/KaimingHe/flax_dev.git
BRANCH=main

salt=`head /dev/urandom | tr -dc a-z0-9 | head -c8`

CONFIG=tpu_vit_base
# JOBNAME=flax/$(date +%Y%m%d_%H%M)_${salt}_${CONFIG}_cjit0.4dbg_lb0.1_cropv4_exwd_initv2_rsinit_dp0.1_cutmixup
JOBNAME=flax/$(date +%Y%m%d_%H%M)_${salt}_${CONFIG}_autoaug_lb0.1_cropv4_exwd_initv2_rsinit_dp0.1_cutmixup


WORKDIR=gs://kmh-gcp/checkpoints/${JOBNAME}
LOGDIR=/home/${USER}/logs/${JOBNAME}
mkdir -p ${LOGDIR}

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

# kill
# gcloud alpha compute tpus tpu-vm ssh ${VM_NAME} --zone europe-west4-a \
#     --worker=all --command "
# sudo pkill python
# source ~/flax_dev/examples/imagenet/run_kill.sh
# sudo lsof -w /dev/accel0
# "

# gcloud alpha compute tpus tpu-vm scp /home/kaiminghe/flax_dev ${VM_NAME}:/home/kaiminghe/flax_dev --recurse --zone europe-west4-a --worker=all

gcloud alpha compute tpus tpu-vm ssh ${VM_NAME} --zone europe-west4-a \
    --worker=all --command "
# git config --global credential.helper store
# git clone -b $BRANCH $REPO

# pip install 'jax[tpu]>=0.2.21' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
# pip install --upgrade clu

# pip3 install torchvision --upgrade
# pip3 install tensorflow-probability
# pip3 install tensorflow_addons

cd ~/flax_dev
git checkout vit
git pull

git rev-parse --short HEAD

# python3 -c 'import jax; print(jax.device_count())'
# python3 -c 'import flax'

cd ~/flax_dev/examples/imagenet
export TFDS_DATA_DIR=gs://kmh-gcp/tensorflow_datasets
python3 main.py \
    --workdir=$WORKDIR \
    --config=configs/$CONFIG.py \
    --config.batch_size=4096 \
    --config.log_every_steps=100 \
" 2>&1 | tee $LOGDIR/finetune.log

echo ${VM_NAME}

