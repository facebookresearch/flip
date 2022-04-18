# VM_NAME=kmh-tpuvm-v3-128-1
VM_NAME=kmh-tpuvm-v3-256-3
echo $VM_NAME

REPO=https://71d519550fe3430ecbf39b70467e9210aed5da69:@github.com/KaimingHe/flax_dev.git
BRANCH=main

# salt=`head /dev/urandom | tr -dc a-z0-9 | head -c8`

seed=0
batch=1024
lr=1e-3
lrd=0.75
ep=50
dp=0.2

vitsize=large
CONFIG=cfg_vit_${vitsize}
source scripts/select_chkpt_${vitsize}.sh

name=`basename ${PRETRAIN_DIR}`

# finetune_pytorch_recipe (ftpy): lb0.1_b0.999_cropv4_exwd_initv2_headinit0.001_tgap_dp_mixup32_cutmix32_noerase_warmlr_minlr_autoaug
JOBNAME='flax/2021-10-26-22-16-05-v3-128-mb4096-epo1600-PMAEp16-ViTLarge-lr1e-4-wd5e-2-warm40-mask0.75-pred8d512-exNB-msaLNmlpLNeLNpLNkBN0-1view-NOrelpos-abspos-clstoken-qkv-NOlayerscale-LNtgt-resume3_convert_pt2jax_finetune/20220418_180021_kmh-tpuvm-v3-256-4_cfg_vit_large_50ep_ftpy_b1024_lr1e-3_lrd0.75_dp0.2_randaugv2_shf512x32_hostbatch_seed0_TorchLoader_DBGbest'
# JOBNAME=flax/${name}_finetune/$(date +%Y%m%d_%H%M%S)_${VM_NAME}_${CONFIG}_${ep}ep_ftpy_b${batch}_lr${lr}_lrd${lrd}_dp${dp}_randaugv2_shf512x32_hostbatch_seed${seed}_TorchLoader_DBGbest

WORKDIR=gs://kmh-gcp/checkpoints/${JOBNAME}
LOGDIR=/home/${USER}/logs/${JOBNAME}
mkdir -p ${LOGDIR}

# source run_init_remote.sh

gcloud alpha compute tpus tpu-vm ssh ${VM_NAME} --zone europe-west4-a \
    --worker=all --command "
cd ~/flax_dev
git pull
git checkout vit.ft.torchloader
git pull
git rev-parse --short HEAD

cd ~/flax_dev
export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=8589934592
export TFDS_DATA_DIR=gs://kmh-gcp/tensorflow_datasets

python3 main.py \
    --workdir=${WORKDIR} \
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
    --config.aug.mix.mixup=True \
    --config.aug.mix.cutmix=True \
    --config.aug.mix.batch_size=32 \
    --config.aug.randerase.on=False \
    --config.aug.autoaug=randaugv2 \
    --config.model.transformer.droppath_rate=${dp} \
    --config.aug.mix.switch_mode=host_batch \
    --config.seed_tf=${seed} \
    --config.seed_jax=${seed} \
    --config.aug.shuffle_buffer_size=16384 \
    --config.model.transformer.torch_qkv=False \
    --config.aug.torchvision=False \
    --config.aug.mix.torchvision=False \

" 2>&1 | tee $LOGDIR/finetune.log

echo ${VM_NAME}