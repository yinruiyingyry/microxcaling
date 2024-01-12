#!/usr/bin/env python

import argparse
import json
import os
import subprocess

model_to_path = {
    'resnet18': 'classification/torchvision',
    'resnet50': 'classification/torchvision',
    'mobilenet_v2': 'classification/torchvision',
    'deit_tiny_patch16_224': 'classification/deit',
    'deit_small_patch16_224': 'classification/deit',
    'facebook/opt-125m': 'language_modeling',
    'facebook/opt-350m': 'language_modeling',
    'facebook/opt-1.3b': 'language_modeling',
    'facebook/opt-2.7b': 'language_modeling',
    'facebook/opt-6.7b': 'language_modeling',
    'gpt2': 'language_modeling',
    'gpt2-medium': 'language_modeling',
    'gpt2-large': 'language_modeling',
    'gpt2-xl': 'language_modeling',
    'meta-llama/Llama-2-7b-hf': 'language_modeling',
}

formats = ['int8', 'fp8_e4m3', 'fp8_e5m2', 'fp6_e2m3', 'fp6_e3m2', 'fp4_e2m1']

def get_args_parser(add_help=True):
  parser = argparse.ArgumentParser(
      'MX direct inference experiments', add_help=False)
  parser.add_argument(
      '--model', default='facebook/opt-125m', type=str, metavar='MODEL')
  return parser

def main(args):
  if args.model not in model_to_path:
    raise ValueError(f'Not a valid model: {args.model}')
  script_path = os.path.join(model_to_path[args.model], 'run_direct_infer.sh')

  # Eval float32 model
  script_args = ['--model', args.model, '--no_mx']
  subprocess.run(['bash', script_path] + script_args)

  for fmt in formats:
    script_args = [
        '--model', args.model, '--format', fmt, '--round_weight', 'nearest',
        '--round_output', 'nearest', '--round_mx_output', 'even'
    ]
    subprocess.run(['bash', script_path] + script_args)

if __name__ == "__main__":
  args = get_args_parser().parse_args()
  print('Used arguments:', args)
  main(args)
