# source run_env.sh

rm -rf tmp

# 4096 / 256 tpus = 128 / 8 tpus  
export TFDS_DATA_DIR=gs://kmh-gcp/tensorflow_datasets
python3 main.py \
    --workdir=./tmp \
    --config=configs/cfg_vit_base.py \
    --config.batch_size=128 \
    --config.log_every_steps=10 \
    --config.num_epochs=0.005 \
    --config.profile_memory=True \
    --config.model.patches.size=\(16,16\) \
    --config.ema=True \
    --config.donate=True \
    --config.pretrain_dir='gs://kmh-gcp/checkpoints/flax/20220331_030004_kmh-tpuvm-v3-128-2_cfg_mae_base_800ep_maeDBG_batch4096_vmap_normpix_sincos_initmaev1'

    # --config.model.classifier='gap' \

    # --config.model.transformer.num_layers=12 \
    # --config.model.hidden_size=768 \

    # --config.model.transformer.num_layers=2 \


# python3 test_profile.py

# pprof -http=localhost:6062 tmp/memory.prof

# PROF_DIR='gs://foo/bar'
# salt=`head /dev/urandom | tr -dc a-z0-9 | head -c8` && TGT_DIR='/tmp/'`basename $PROF_DIR`'_memory_'${salt}'.prof' && gsu cp $PROF_DIR/memory.prof $TGT_DIR && echo $TGT_DIR
# pprof -http=localhost:6062 $TGT_DIR
