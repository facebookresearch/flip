from typing import Tuple, Any

from absl import logging
from termcolor import colored

import tree as nest

import jax
import flax
from flax.training import checkpoints
import jax.numpy as jnp
import numpy as np

import jax.tree_util as tu

import torch
import os

from utils import checkpoint_util


def convert_to_pytorch(state, pretrain_dir, config):
  seperate_qkv = config.model.transformer.seperate_qkv  # this means JAX and PyTorch models are consisent

  state = checkpoint_util.load_from_pretrain(state, pretrain_dir)  # restore from JAX checkpoint
  state_params = flax.core.frozen_dict.unfreeze(state.params)
  state_params.pop('head')  # remove head
  state_params = flax.core.frozen_dict.freeze(state_params)

  # create a list of named params
  names = filter_parameters(state_params, get_name)
  params_list, params_tree = tu.tree_flatten(state_params)
  names_list, names_tree = tu.tree_flatten(names)
  named_params = dict((x, y) for x, y in zip(names_list, params_list))

  pytorch_model_dir='gs://kmh-gcp/from_pytorch/template_pretrained_lastnorm_tf2pt.large.pth'  # this is a template
  os.system('gsutil cp {} ./tmp/.'.format(pytorch_model_dir))

  checkpoint = torch.load(os.path.join('./tmp', os.path.basename(pytorch_model_dir)), map_location='cpu')
  checkpoint = checkpoint['model']
  
  checkpoint_revised = checkpoint if seperate_qkv else revise_split_qkv(checkpoint)

  converted_checkpoint = convert_names_and_shapes_j2p(checkpoint_revised, named_params, seperate_qkv)
  converted_checkpoint = converted_checkpoint if seperate_qkv else revise_merge_qkv(converted_checkpoint)

  new_keys = set(converted_checkpoint.keys()) - set(checkpoint.keys())
  logging.info('New keys: {}'.format(str(new_keys)))
  missing_keys = set(checkpoint.keys()) - set(converted_checkpoint.keys())
  logging.info('Missing keys: {}'.format(str(missing_keys)))

  # save file
  basename = os.path.basename(pretrain_dir)
  torch.save(converted_checkpoint, './tmp/converted_jax2pt.pth')
  output_dir = os.path.join('gs://kmh-gcp/to_pytorch', basename, 'converted_jax2pt.pth')
  cmd = 'gsutil cp ./tmp/converted_jax2pt.pth {}'.format(output_dir)
  os.system(cmd)

  final_dir = os.path.join('/checkpoint/kaiminghe/converted', basename, 'converted_jax2pt.pth')
  logging.info('Copy this and run in devfair:')
  print('gsutil cp {} {}'.format(output_dir, final_dir))


def convert_from_pytorch(state, pretrain_dir):
  state_params = flax.core.frozen_dict.unfreeze(state.params)
  state_params.pop('head')  # remove head
  state_params = flax.core.frozen_dict.freeze(state_params)

  # create a list of named params
  names = filter_parameters(state_params, get_name)
  params_list, params_tree = tu.tree_flatten(state_params)
  names_list, names_tree = tu.tree_flatten(names)
  named_params = dict((x, y) for x, y in zip(names_list, params_list))

  os.system('gsutil cp {} ./tmp/.'.format(pretrain_dir))

  checkpoint = torch.load(os.path.join('./tmp', os.path.basename(pretrain_dir)), map_location='cpu')
  checkpoint = checkpoint['model']
  checkpoint = revise_split_qkv(checkpoint)  # split qkv to match the JAX format

  converted_named_params = convert_names_and_shapes_p2j(checkpoint, named_params)

  converted_params_list = []
  for name in names_list:
    converted_params_list.append(jnp.asarray(converted_named_params[name]))

  converted_params = tu.tree_unflatten(params_tree, converted_params_list)
  converted_params = flax.core.frozen_dict.freeze(converted_params)
  
  # sanity
  verify = tu.tree_map(lambda x, y: (x.shape == y.shape),
      converted_params, state_params)
  verify = tu.tree_leaves(verify)
  assert jnp.all(jnp.array(verify)).item()

  state = state.replace(params=converted_params)

  # save file
  output_dir = os.path.dirname(pretrain_dir)
  output_dir = output_dir + '_convert_pt2jax'
  checkpoints.save_checkpoint(output_dir, state, step=0, overwrite=False)

  return
  

def convert_names_and_shapes_p2j(checkpoint, named_params):
  converted_named_params = {}
  for name_pt in checkpoint:
    p_pt = checkpoint[name_pt].clone()
    shape_pt = tuple(p_pt.shape)

    name_jx = convert_name(name_pt)
    if name_jx in named_params:
      shape_jx = tuple(named_params[name_jx].shape)
      if len(shape_pt) == 1 and len(shape_jx) == 1:  # 1-d tensors
          assert shape_pt == shape_jx
      elif len(shape_pt) == 4 and len(shape_jx) == 4:  # patch_embed
          p_pt = torch.einsum('nchw->hwcn', p_pt)
          assert tuple(p_pt.shape) == shape_jx
      elif len(shape_pt) == 3 and len(shape_jx) == 3:  # pos_embed
          assert shape_pt == shape_jx
      elif len(shape_pt) == 2 and len(shape_jx) == 2:  # mlp
          p_pt = torch.einsum('nc->cn', p_pt)
          assert tuple(p_pt.shape) == shape_jx
      elif len(shape_pt) == 2 and len(shape_jx) == 3 and '.out.kernel' in name_jx:  # msa, out.kernel
          p_pt = torch.einsum('nc->cn', p_pt)  # n: out_filters; c: in_filters
          assert shape_jx[-1] == p_pt.shape[-1]
          p_pt = p_pt.view(shape_jx)
      elif len(shape_pt) == 2 and len(shape_jx) == 3 and '.out.kernel' not in name_jx:  # msa, q, k, v
          p_pt = torch.einsum('nc->cn', p_pt)  # n: out_filters; c: in_filters
          assert shape_jx[0] == p_pt.shape[0]
          p_pt = p_pt.view(shape_jx)
      elif len(shape_pt) == 1 and len(shape_jx) == 2:  # q, k, v bias
          p_pt = p_pt.view(shape_jx)

      # now, p_pt is the expected tensor
      assert tuple(p_pt.shape) == shape_jx
      print('{:32s}:{:20s} => {:80s}:{:20s}'.format(name_pt, str(shape_pt), name_jx, str(shape_jx)))
      converted_named_params[name_jx] = p_pt
    else:
      print(colored('Not converted: {} => {}'.format(name_pt, name_jx), 'red'))

  return converted_named_params


def convert_names_and_shapes_j2p(checkpoint, named_params, seperate_qkv):
  converted_checkpoint = {}
  for name_pt in checkpoint:
    p_pt = checkpoint[name_pt].clone()
    shape_pt = tuple(p_pt.shape)

    name_jx = convert_name(name_pt, seperate_qkv)
    if name_jx in named_params:
      p_jx = named_params[name_jx]
      shape_jx = tuple(p_jx.shape)
      if len(shape_pt) == 1 and len(shape_jx) == 1:  # 1-d tensors
          assert shape_pt == shape_jx
      elif len(shape_pt) == 4 and len(shape_jx) == 4:  # patch_embed
          p_jx = jnp.einsum('hwcn->nchw', p_jx)
          assert tuple(p_jx.shape) == shape_pt
      elif len(shape_pt) == 3 and len(shape_jx) == 3:  # pos_embed
          assert shape_pt == shape_jx
      elif len(shape_pt) == 2 and len(shape_jx) == 2:  # mlp
          p_jx = jnp.einsum('cn->nc', p_jx)
          assert tuple(p_jx.shape) == shape_pt
      elif len(shape_pt) == 2 and len(shape_jx) == 3 and '.out.kernel' in name_jx:  # msa, out.kernel
          assert shape_jx[-1] == p_pt.shape[0]
          p_jx = jnp.reshape(p_jx, [-1, p_jx.shape[-1]])
          p_jx = jnp.einsum('cn->nc', p_jx)
      elif len(shape_pt) == 2 and len(shape_jx) == 3 and '.out.kernel' not in name_jx:  # msa, q, k, v
          assert shape_jx[0] == p_pt.shape[-1]
          p_jx = jnp.reshape(p_jx, [p_jx.shape[0], -1,])
          p_jx = jnp.einsum('cn->nc', p_jx)
      elif len(shape_pt) == 1 and len(shape_jx) == 2:  # q, k, v bias
          p_jx = jnp.reshape(p_jx, shape_pt)

      # now, p_pt is the expected tensor
      assert tuple(shape_pt) == p_jx.shape
      print('{:32s}:{:20s} <= {:80s}:{:20s}'.format(name_pt, str(shape_pt), name_jx, str(shape_jx)))
      converted_checkpoint[name_pt] = torch.tensor(np.array(p_jx))
    else:
      print(colored('Not converted: {} => {}'.format(name_pt, name_jx), 'red'))

  return converted_checkpoint


def revise_merge_qkv(checkpoint):
  # handle q, k, v
  checkpoint_revised = {}
  for name_pt in checkpoint:
    p = checkpoint[name_pt]
    if 'attn.q.weight' in name_pt:
      q = checkpoint[name_pt]
      k = checkpoint[name_pt.replace('.q.weight', '.k.weight')]
      v = checkpoint[name_pt.replace('.q.weight', '.v.weight')]
      qkv = torch.concat([q, k, v], dim=0)
      checkpoint_revised[name_pt.replace('.q.weight', '.qkv.weight')] = qkv
    elif 'attn.k.weight' in name_pt or 'attn.v.weight' in name_pt:
      pass      
    else:
      checkpoint_revised[name_pt] = p

  del checkpoint
  return checkpoint_revised


def revise_split_qkv(checkpoint):
  # handle q, k, v
  checkpoint_revised = {}
  for name_pt in checkpoint:
    p = checkpoint[name_pt].clone()
    if 'attn.qkv.weight' in name_pt:
      q, k, v = torch.split(p, p.shape[0] // 3, dim=0)
      checkpoint_revised[name_pt.replace('.qkv.', '.q.')] = q
      checkpoint_revised[name_pt.replace('.qkv.', '.k.')] = k
      checkpoint_revised[name_pt.replace('.qkv.', '.v.')] = v
    else:
      checkpoint_revised[name_pt] = p

    if 'attn.q_bias' in name_pt:  # add k_bias
      assert name_pt.replace('.q_bias', '.k_bias') not in checkpoint
      checkpoint_revised[name_pt.replace('.q_bias', '.k_bias')] = torch.zeros_like(p)

  del checkpoint
  return checkpoint_revised


def convert_name(name, seperate_qkv):
  if seperate_qkv:
    msa_prefix = 'MultiHeadDotProductAttentionQKV_0'
  else:
    msa_prefix = 'MultiHeadDotProductAttention_0'
  # convert:
  if name == 'cls_token':
    name_jx = 'cls'
  elif name == 'pos_embed':
    name_jx = 'posembed_encoder.pos_embedding'
  elif name == 'patch_embed.proj.weight':
    name_jx = 'embedding.kernel'
  elif name == 'patch_embed.proj.bias':
    name_jx = 'embedding.bias'
  elif name.startswith('norm.weight'):
    name_jx = 'Transformer.encoder_norm.bias'
  elif name.startswith('norm.bias'):
    name_jx = 'Transformer.encoder_norm.scale'
  elif name == 'head.weight':
    name_jx = 'head.kernel'
  elif name == 'head.bias':
    name_jx = 'head.bias'
  elif name.startswith('blocks.'):
    name_suffix = name[len('blocks.'):]  # remove 'block.'
    blk_idx = name_suffix[:name_suffix.find('.')]  # get 11 from '11.mlp.fc2.weight'
    blk_idx = int(blk_idx)
    
    name_jx = 'Transformer.encoderblock_{:02d}.'.format(blk_idx)

    # MSA
    if 'norm1.weight' in name:
      name_jx += 'LayerNorm_0.scale'
    elif 'norm1.bias' in name:
      name_jx += 'LayerNorm_0.bias'
    elif 'attn.proj.weight' in name:
      name_jx += msa_prefix + '.out.kernel'
    elif 'attn.proj.bias' in name:
      name_jx += msa_prefix + '.out.bias'
    elif 'attn.q.weight' in name:
      name_jx += msa_prefix + '.query.kernel'
    elif 'attn.k.weight' in name:
      name_jx += msa_prefix + '.key.kernel'
    elif 'attn.v.weight' in name:
      name_jx += msa_prefix + '.value.kernel'
    elif 'attn.qkv.weight' in name:
      name_jx += msa_prefix + '.qkv.kernel'
    elif 'attn.q_bias' in name:
      name_jx += msa_prefix + '.query.bias' if not seperate_qkv else msa_prefix + '.q_bias'
    elif 'attn.k_bias' in name:
      name_jx += msa_prefix + '.key.bias'
    elif 'attn.v_bias' in name:
      name_jx += msa_prefix + '.value.bias' if not seperate_qkv else msa_prefix + '.v_bias'
    # MLP
    elif 'norm2.weight' in name:
      name_jx += 'LayerNorm_1.scale'
    elif 'norm2.bias' in name:
      name_jx += 'LayerNorm_1.bias'
    elif 'mlp.fc1.weight' in name:
      name_jx += 'MlpBlock_0.Dense_0.kernel'
    elif 'mlp.fc1.bias' in name:
      name_jx += 'MlpBlock_0.Dense_0.bias'
    elif 'mlp.fc2.weight' in name:
      name_jx += 'MlpBlock_0.Dense_1.kernel'
    elif 'mlp.fc2.bias' in name:
      name_jx += 'MlpBlock_0.Dense_1.bias'
  else:
    return None
    # raise NotImplementedError
  return name_jx


# ---------------------------------------------------------
# the tree functions:
# ---------------------------------------------------------
def get_name(path: Tuple[Any], val: jnp.ndarray):
  del val
  layer_name = '.'.join(path)
  return layer_name


def filter_parameters(params, filter_fn):
  """Filter the params based on filter_fn."""
  params_to_filter = nest.map_structure_with_path(filter_fn, params)
  return params_to_filter