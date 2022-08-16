echo 'code dir: '$STAGEDIR

# seed=0
batch=4096
lr=1e-4
ep=800

mask=0.75
mask_txt=0.25

partitions=1

rescale=1.0

vitsize=large
CONFIG=cfg_mae_${vitsize}

# _normpix_exwd_NOsplit_fastsave
JOBNAME=flax/$(date +%Y%m%d_%H%M%S)_maet5x_${VM_NAME}_${CONFIG}_${ep}ep_b${batch}_lr${lr}_mk${mask}txt${mask_txt}_s${seed}_p${partitions}st_re${rescale}_laion_a0.5_sanity_twoMAE_twoCross
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
    --config.num_epochs=${ep} \
    --config.learning_rate=${lr} \
    --config.profile_memory=True \
    --config.model.model_img.transformer.rescale_init=${rescale} \
    --config.model.model_img.norm_pix_loss=True \
    --config.model.model_img.sincos=True \
    --config.model.model_img.mask_ratio=${mask} \
    --config.model.model_txt.mask_ratio=${mask_txt} \
    --config.seed_tf=${seed} \
    --config.seed_jax=${seed} \
    --config.seed_pt=${seed} \
    --config.partitioning.num_partitions=${partitions} \
    --config.opt_type=adamw \
    --config.opt_mu_dtype=float32 \
    --config.partitioning.partition_states=False \
    --config.resume_dir=${RESUME} \
    --config.aug.area_range=\(0.5\,1.0\) \
    --config.model.model_txt.decoder.cross_attention=True \
    --config.model.model_img.decoder.cross_attention=True \
2>&1 | tee -a $LOGDIR/finetune_\$SSH_ID.log
" 2>&1 | tee -a $LOGDIR/finetune.log

echo ${VM_NAME}