# VM_NAME=kmh-tpuvm-v3-128
VM_NAME=kmh-tpuvm-v3-256-3
echo $VM_NAME
REPO=https://71d519550fe3430ecbf39b70467e9210aed5da69:@github.com/KaimingHe/flax_dev.git
BRANCH=main

salt=`head /dev/urandom | tr -dc a-z0-9 | head -c8`

ep=300
CONFIG=tpu_vit_base
# JOBNAME=flax/$(date +%Y%m%d_%H%M)_${salt}_${CONFIG}_cjit0.4dbg_lb0.1_cropv4_exwd_initv2_rsinit_dp0.1_cutmixup
JOBNAME=flax/$(date +%Y%m%d_%H%M)_${salt}_${VM_NAME}_${CONFIG}_${ep}ep_autoaug_lb0.1_cropv4_exwd_initv2_rsinit_dp0.1_cutmixup


WORKDIR=gs://kmh-gcp/checkpoints/${JOBNAME}
LOGDIR=/home/${USER}/logs/${JOBNAME}
mkdir -p ${LOGDIR}

# source run_init_remote.sh

# check libraries
gcloud alpha compute tpus tpu-vm ssh ${VM_NAME} --zone europe-west4-a \
    --worker=0 --command "
pip3 list | grep jax
pip3 list | grep flax
pip3 list | grep tensorflow
"

gcloud alpha compute tpus tpu-vm ssh ${VM_NAME} --zone europe-west4-a \
    --worker=all --command "
cd ~/flax_dev
git checkout vit
git pull
git rev-parse --short HEAD

cd ~/flax_dev/examples/imagenet
export TFDS_DATA_DIR=gs://kmh-gcp/tensorflow_datasets
python3 main.py \
    --workdir=$WORKDIR \
    --config=configs/$CONFIG.py \
    --config.batch_size=4096 \
    --config.log_every_steps=100 \
    --config.num_epochs=${ep} \
" 2>&1 | tee $LOGDIR/finetune.log

echo ${VM_NAME}

