# source run_env.sh

rm -rf tmp

# 4096 / 256 tpus = 128 / 8 tpus  
export TFDS_DATA_DIR=gs://kmh-gcp/tensorflow_datasets
python3 main.py \
    --workdir=./tmp \
    --config=configs/cfg_vit_huge.py \
    --config.batch_size=256 \
    --config.log_every_steps=10 \
    --config.num_epochs=0.005 \
    --config.profile_memory=True \
    --config.model.patches.size=\(16,16\) \
    --config.ema=False

    # --config.model.transformer.num_layers=2 \


# python3 test_profile.py

# pprof -http=localhost:6062 tmp/memory.prof

# PROF_DIR='gs://foo/bar'
# salt=`head /dev/urandom | tr -dc a-z0-9 | head -c8` && TGT_DIR='/tmp/'`basename $PROF_DIR`'_memory_'${salt}'.prof' && gsu cp $PROF_DIR/memory.prof $TGT_DIR && echo $TGT_DIR
# pprof -http=localhost:6062 $TGT_DIR
