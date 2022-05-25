echo 'code dir: '$STAGEDIR

# seed=0
batch=4096
lr=1e-4
wd=0.3
lrd=1.0
ep=50
warm=20
dp=0.2
beta2=0.95

mask=0.75

partitions=1

vitsize=large
CONFIG=cfg_mae_${vitsize}


JOBNAME=flax/$(date +%Y%m%d_%H%M%S)_maedbg_${VM_NAME}_${CONFIG}_${ep}ep_fttl_b${batch}_wd${wd}_lr${lr}_mk${mask}_lrd${lrd}_dp${dp}_warm${warm}_s${seed}_beta${beta2}_p${partitions}_speeddbg_decoder

WORKDIR=gs://kmh-gcp/checkpoints/${JOBNAME}
LOGDIR=/kmh_data/logs/${JOBNAME}
mkdir -p ${LOGDIR}
chmod 777 ${LOGDIR}

# source run_init_remote.sh

gcloud alpha compute tpus tpu-vm ssh ${VM_NAME} --zone europe-west4-a \
    --worker=all --command "
cd $STAGEDIR
git config --global --add safe.directory $STAGEDIR

echo Current commit: $(git show -s --format=%h)
echo Current dir: $(pwd)

export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=8589934592
export TFDS_DATA_DIR=gs://kmh-gcp/tensorflow_datasets

source run_get_ssh_id.sh

python3 main.py \
    --workdir=${WORKDIR} \
    --config=configs/$CONFIG.py \
    --config.batch_size=${batch} \
    --config.learning_rate=${lr} \
    --config.learning_rate_decay=${lrd} \
    --config.opt.weight_decay=${wd} \
    --config.opt.b2=${beta2} \
    --config.warmup_epochs=${warm} \
    --config.log_every_steps=100 \
    --config.num_epochs=${ep} \
    --config.save_every_epochs=50 \
    --config.profile_memory=True \
    --config.donate=True \
    --config.init_backend=tpu \
    --config.model.mask_ratio=${mask} \
    --config.model.transformer.droppath_rate=${dp} \
    --config.seed_tf=${seed} \
    --config.seed_jax=${seed} \
    --config.seed_pt=${seed} \
    --config.model.classifier=token \
    --config.partitioning.num_partitions=${partitions} \
    --config.opt_type=adamw \
    --config.opt_mu_dtype=float32 \
2>&1 | tee $LOGDIR/finetune_\$SSH_ID.log
" 2>&1 | tee $LOGDIR/finetune.log

echo ${VM_NAME}