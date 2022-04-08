# VM_NAME=kmh-tpuvm-v3-128-1
VM_NAME=kmh-tpuvm-v3-256-3
echo $VM_NAME
REPO=https://71d519550fe3430ecbf39b70467e9210aed5da69:@github.com/KaimingHe/flax_dev.git
BRANCH=main

# salt=`head /dev/urandom | tr -dc a-z0-9 | head -c8`

batch=1024
lr=1e-3
lrd=0.75
ep=50
cls='tgap'

head_init=0.001

vitsize=large
CONFIG=cfg_vit_${vitsize}
source scripts/select_chkpt_${vitsize}.sh

name=`basename ${PRETRAIN_DIR}`

# pytorch_recipe (pyre): _autoaug_lb0.1_cropv4_exwd_initv2_rsinit_dp0.1_cutmixup_minlr
JOBNAME=flax/${name}_finetune/$(date +%Y%m%d_%H%M%S)_${VM_NAME}_${CONFIG}_${ep}ep_FT_b${batch}_lr${lr}_lrd${lrd}_${cls}_hinit${head_init}_b0.999_32mixup_NOcutmix_NOaa_NOerase_warmlr_NOencnorm_shf512b_fullevsp

WORKDIR=gs://kmh-gcp/checkpoints/${JOBNAME}
LOGDIR=/home/${USER}/logs/${JOBNAME}
mkdir -p ${LOGDIR}

# source run_init_remote.sh

# check libraries
# gcloud alpha compute tpus tpu-vm ssh ${VM_NAME} --zone europe-west4-a \
#     --worker=0 --command "
# pip3 list | grep jax
# pip3 list | grep flax
# pip3 list | grep tensorflow
# "

gcloud alpha compute tpus tpu-vm ssh ${VM_NAME} --zone europe-west4-a \
    --worker=all --command "
cd ~/flax_dev
git pull
git checkout vit.ft.subtleties
git pull
git rev-parse --short HEAD

# pip3 list | grep 'jax\|flax\|tensorflow '

cd ~/flax_dev
export TFDS_DATA_DIR=gs://kmh-gcp/tensorflow_datasets
python3 main.py \
    --workdir=$WORKDIR \
    --config=configs/$CONFIG.py \
    --config.pretrain_dir=${PRETRAIN_DIR} \
    --config.batch_size=${batch} \
    --config.learning_rate=${lr} \
    --config.learning_rate_decay=${lrd} \
    --config.log_every_steps=100 \
    --config.num_epochs=${ep} \
    --config.ema=False \
    --config.save_every_epochs=10 \
    --config.profile_memory=True \
    --config.donate=True \
    --config.init_backend=tpu \
    --config.model.classifier=${cls} \
    --config.rescale_head_init=${head_init} \
    --config.aug.mix.mixup=True \
    --config.aug.mix.cutmix=False \
    --config.aug.mix.batch_size=32 \
    --config.aug.autoaug=None \
    --config.aug.randerase.on=False \
    --config.warmup_abs_lr=1e-6 \
" 2>&1 | tee $LOGDIR/finetune.log

echo ${VM_NAME}
