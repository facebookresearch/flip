## Installation

We train FLIP models on TPU by default and use `tpu-vm-base` TPU software version. We use python 3.8.10 and install the following dependencies:

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

For ImageNet-1K, we follow the [link](https://github.com/google/flax/tree/main/examples/imagenet#preparing-the-dataset) to prepare the dataset.

#### LAION datasets

We download the [LAION-400M dataset](https://laion.ai/blog/laion-400-open-dataset/) and the [LAION-2B-en dataset](https://laion.ai/blog/laion-5b/) (the English subset of LAION-5B) and prepare them into TFRecord files using the [img2dataset](https://github.com/rom1504/img2dataset) tool.

Specifically, we first download the metadata `*.parquet` files (containing text and image URLs) from Hugging Face datasets for [LAION-400M](https://huggingface.co/datasets/laion/laion400m/tree/main) and [LAION-2B-en](https://huggingface.co/datasets/laion/laion2B-en/tree/main). Then we use `img2dataset` to download the images, resize them into a short size of 480 pixels, and pack them into TFRecord files as follows. (Here we download them into Google Cloud Storage buckets `OUTPUT_DIR`, but you can also use a local directory for `OUTPUT_DIR` as long as it has enough disk space.)

LAION-400M:
```bash
pip3 install img2dataset tensorflow tensorflow-io

META_DATA=path/to/laion400m-meta  # download from https://huggingface.co/datasets/laion/laion400m/tree/main
OUTPUT_DIR=gs://your-gcs-bucket/laion-400m/tfrecord_dataset_img480

img2dataset --url_list $META_DATA --input_format "parquet" \
  --url_col "URL" --caption_col "TEXT" --output_format tfrecord \
  --output_folder $OUTPUT_DIR \
  --processes_count 32 --thread_count 256 --image_size 480 --resize_mode keep_ratio  \
  --save_additional_columns '["NSFW","similarity","LICENSE"]'
```

LAION-2B-en:
```bash
pip3 install img2dataset tensorflow tensorflow-io

META_DATA=path/to/laion2b-en-meta  # download from https://huggingface.co/datasets/laion/laion2B-en/tree/main
OUTPUT_DIR=gs://your-gcs-bucket/laion-2b-en/tfrecord_dataset_img480

img2dataset --url_list $META_DATA --input_format "parquet" \
  --url_col "URL" --caption_col "TEXT" --output_format tfrecord \
  --output_folder $OUTPUT_DIR \
  --processes_count 32 --thread_count 256 --image_size 480 --resize_mode keep_ratio  \
  --save_additional_columns '["NSFW","similarity","LICENSE"]'
```

More details can be found in the img2dataset examples for [LAION-400M](https://github.com/rom1504/img2dataset/blob/main/dataset_examples/laion400m.md) and [LAION-2B-en](https://github.com/rom1504/img2dataset/blob/main/dataset_examples/laion5B.md).
