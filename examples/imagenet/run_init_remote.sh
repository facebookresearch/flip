REPO=https://71d519550fe3430ecbf39b70467e9210aed5da69:@github.com/KaimingHe/flax_dev.git
BRANCH=main

gcloud alpha compute tpus tpu-vm ssh ${VM_NAME} --zone europe-west4-a \
    --worker=0 --command "
# git config --global credential.helper store
# git clone -b $BRANCH $REPO

# pip install 'jax[tpu]==0.2.21' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
# pip install jaxlib==0.1.70
# pip install flax==0.3.6
# pip install --upgrade clu

# pip3 install torchvision --upgrade
# pip3 install tensorflow-probability
# pip3 install tensorflow_addons

# cd ~/flax_dev
# git checkout vit
# git pull

# git rev-parse --short HEAD

# python3 -c 'import jax; print(jax.device_count())'
# python3 -c 'import flax'

# pip3 list | grep jax
# pip3 list | grep flax
pip3 list | grep tensorflow
"