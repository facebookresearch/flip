## Installation

We train FLIP models on TPU by default. We use python 3.8.10 and install the following dependencies:

```
sudo pip3 install clu==0.0.7
sudo pip3 install timm==0.4.12
sudo pip3 install tensorflow_addons
sudo pip3 install jax[tpu]==0.3.15 jaxlib==0.3.15 flax==0.5.3 -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
sudo pip3 install absl-py cached_property gin-config numpy orbax seqio-nightly tensorstore

sudo pip3 uninstall -y tensorflow
sudo pip3 uninstall -y tensorflow  # do it twice in case of any duplicated installation
sudo pip3 uninstall -y tf-nightly
sudo pip3 install tensorflow==2.8.0
sudo pip3 install tensorflow-text==2.7.3
sudo pip3 install pandas==1.4.4
sudo pip3 install transformers==4.21.3
sudo pip3 install numpy==1.23.3
```

## Dataset preparation

We train on LAION datasets and evalute on ImageNet-1K dataset.

#### ImageNet-1K

For ImageNet-1K, we following the [link](https://github.com/google/flax/tree/main/examples/imagenet#preparing-the-dataset) to prepare the dataset.


#### LAION datasets