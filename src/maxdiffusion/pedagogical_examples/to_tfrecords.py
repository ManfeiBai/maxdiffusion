"""
 Copyright 2024 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

"""
Example file of how to prepare tfrecords with latents and hidden_states preprocessed.
1. Download the dataset as instructed here https://github.com/mlcommons/training/tree/master/stable_diffusion#the-datasets
2. Mount gcs bucket using gcs fuse.
3. Create 2 directories inside gcs bucket to store the extracted files and created tfrecords.
3. Run this file:

python src/maxdiffusion/pedagogical_examples/to_tfrecords.py \
  src/maxdiffusion/configs/base_2_base.yml \
  attention=dot_product \
  data_files_pattern=/home/jfacevedo/gcsfuse/laion400m/raw_data/webdataset-moments-filtered/*.tar \
  extracted_files_dir=/tmp/extracted \
  tfrecords_dir=/home/jfacevedo/gcsfuse/laion400m/processed/laion400m_tfrec \
  run_name=test \
  base_output_directory=gs://jfacevedo-maxdiffusion/training_results/
"""

import os
import glob
import functools
from absl import app
from typing import Sequence

import tensorflow as tf
import tarfile
import tensorflow_datasets as tfds
import numpy as np
import jax
import jax.numpy as jnp

from maxdiffusion import (
  FlaxStableDiffusionPipeline,
  pyconfig,
  max_utils
)

from maxdiffusion.models.vae_flax import FlaxDiagonalGaussianDistribution

dl_manager = tfds.download.DownloadManager(download_dir="/tmp")
tmp_dataset = "dataset"

def delete_files(path):
  files = glob.glob(path+"/*")
  for f in files:
      os.remove(f)

def tokenize_captions(caption, pipeline, p_encode):
   text_inputs = pipeline.tokenizer([caption],
                                    max_length=pipeline.tokenizer.model_max_length,
                                    padding="max_length",
                                    truncation=True)
   hidden_states = p_encode(np.stack(text_inputs.input_ids))
   hidden_states = jnp.squeeze(hidden_states)
   return hidden_states

def image_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()])
    )

def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.numpy()]))


def float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def float_feature_list(value):
    """Returns a list of float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def create_example(latent, hidden_states):
    latent = tf.io.serialize_tensor(latent)
    hidden_states = tf.io.serialize_tensor(hidden_states)
    feature = {
        "latents": bytes_feature(latent),
        "hidden_states": bytes_feature(hidden_states),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()

def generate_dataset(config, pipeline, p_encode, vae):

  tfrecords_dir=config.tfrecords_dir
  extracted_files_dir=config.extracted_files_dir

  if not os.path.exists(tfrecords_dir):
    os.makedirs(tfrecords_dir)  # creating TFRecords output folder

  tf_rec_num = 0
  filenames = tf.io.gfile.glob(config.data_files_pattern)
  no_records_per_shard = config.no_records_per_shard
  global_record_count = 0
  writer = tf.io.TFRecordWriter(
    tfrecords_dir + "/file_%.2i-%i.tfrec" % (tf_rec_num, (global_record_count + no_records_per_shard)))
  rng = jax.random.PRNGKey(0)
  for filename in filenames:
    tmp_file = dl_manager.download(filename)
    file = tarfile.open(tmp_file)
    file.extractall(extracted_files_dir, filter='data')
    extracted_filenames = tf.io.gfile.glob(f"{extracted_files_dir}/*.npy")
    shard_record_count = 0
    for moments_file in extracted_filenames:
      moment = np.load(moments_file)
      latent = moments_to_latents(moment, rng, vae)
      rng, _ = jax.random.split(rng)
      caption_file = moments_file.split(".")[0] + ".txt"
      with open(caption_file, "r") as f:
        caption = f.read()
      hidden_states = np.array(tokenize_captions(caption, pipeline, p_encode))
      example = create_example(latent, hidden_states)
      writer.write(example)
      shard_record_count+=1
      global_record_count+=1
      if shard_record_count >= no_records_per_shard:
        writer.close()
        writer = tf.io.TFRecordWriter(
          tfrecords_dir + "/file_%.2i-%i.tfrec" % (tf_rec_num, (global_record_count + no_records_per_shard)))
        shard_record_count = 0
    tf_rec_num+=1
    delete_files(tmp_dataset)
    os.remove(tmp_file)

def encode(input_ids, text_encoder, text_encoder_params):
  return text_encoder(
    input_ids,
    params=text_encoder_params,
    train=False
  )[0]

def moments_to_latents(moments, rng, vae):
  moments = jnp.transpose(moments, (0, 2, 3, 1))
  posterior = FlaxDiagonalGaussianDistribution(moments)
  latents = posterior.sample(rng)
  # (NHWC) -> (NCHW)
  latents = jnp.transpose(latents, (0, 3, 1, 2))
  latents = latents * vae.config.scaling_factor
  latents = jnp.squeeze(latents)
  return latents

def run(config):

  weight_dtype = max_utils.get_dtype(config)
  pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
    config.pretrained_model_name_or_path,revision=config.revision, dtype=weight_dtype,
    safety_checker=None, feature_extractor=None,
    split_head_dim=config.split_head_dim, from_pt=config.from_pt,
    attention_kernel=config.attention, flash_block_sizes=None,
    mesh=None
  )

  # The dataset contains moments (look at vae_flax.py ln. 917) in B,C,S,S
  # and they are transposed to B,S,S,C then fully encoded as latents.
  p_encode = jax.jit(functools.partial(encode,
                                           text_encoder=pipeline.text_encoder,
                                           text_encoder_params=params["text_encoder"]))

  generate_dataset(config, pipeline, p_encode, pipeline.vae)

def main(argv: Sequence[str]) -> None:
   pyconfig.initialize(argv)
   run(pyconfig.config)

if __name__ == "__main__":
    app.run(main)
