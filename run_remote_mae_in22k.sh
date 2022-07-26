echo 'code dir: '$STAGEDIR

# seed=0
batch=4096
lr=1e-4
ep=160

mask=0.75

partitions=1

rescale=1.0

vitsize=huge_p14
CONFIG=cfg_mae_${vitsize}


JOBNAME=flax/$(date +%Y%m%d_%H%M%S)_maet5x_${VM_NAME}_IN22K_${CONFIG}_${ep}ep_b${batch}_lr${lr}_mk${mask}_s${seed}_p${partitions}_re${rescale}_normpix_exwd_NOsplit_fastsave
RESUME=''

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

export GOOGLE_APPLICATION_CREDENTIALS=~/gcp_credential.json
export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=8589934592
export TFDS_DATA_DIR=gs://kmh-gcp/tensorflow_datasets

source run_get_ssh_id.sh

python3 main.py \
    --workdir=${WORKDIR} \
    --config=configs/$CONFIG.py \
    --config.batch_size=${batch} \
    --config.log_every_steps=100 \
    --config.save_every_epochs=5 \
    --config.warmup_epochs=4.0 \
    --config.num_epochs=${ep} \
    --config.learning_rate=${lr} \
    --config.model.transformer.rescale_init=${rescale} \
    --config.profile_memory=True \
    --config.model.norm_pix_loss=True \
    --config.model.sincos=True \
    --config.model.mask_ratio=${mask} \
    --config.seed_tf=${seed} \
    --config.seed_jax=${seed} \
    --config.seed_pt=${seed} \
    --config.partitioning.num_partitions=${partitions} \
    --config.opt_type=adamw \
    --config.opt_mu_dtype=float32 \
    --config.partitioning.partition_states=False \
    --config.model.visualize=False \
    --config.resume_dir=${RESUME} \
    --config.torchload.data_dir=/datasets/imagenet-22k \
2>&1 | tee -a $LOGDIR/finetune_\$SSH_ID.log
" 2>&1 | tee -a $LOGDIR/finetune.log

echo ${VM_NAME}