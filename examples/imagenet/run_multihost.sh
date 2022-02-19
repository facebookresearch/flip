REPO=https://github.com/google/flax
BRANCH=main
WORKDIR=gs://kmh-gcp/checkpoints/flax/examples/imagenet/$(date +%Y%m%d_%H%M)

gcloud alpha compute tpus tpu-vm ssh kmh-tpuvm-v3-128 --zone europe-west4-a \
    --worker=all --command "
pip install 'jax[tpu]>=0.2.21' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html &&
pip install --user git+$REPO.git &&
git clone --depth=1 -b $BRANCH $REPO
"