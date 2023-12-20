import argparse
import os
import numpy as np
import time
import functools

import torch
import torch.nn.functional as F

from torch import nn
from torchvision import transforms
from torchvision import datasets
from torchvision.models import resnet18, resnet50, mobilenet_v2

import mx
from mx import add_mx_args, get_mx_specs
from mx import mx_mapping
from mx import simd_ops
from mx import specs

from auto_module import modules as amm

#mx_classes = [
#    mx.Linear,
#    mx.Conv2d,
#    simd_ops.SIMDAddModule,
#    mx.BatchNorm2d,
#    mx.ReLU,
#    mx.ReLU6,
#    mx.Softmax,
#    mx.AdaptiveAvgPool2d,
#]
#
#nn.Linear = mx_classes[0]
#nn.Conv2d = mx_classes[1]
#amm.Add = mx_classes[2]
#nn.BatchNorm2d = mx_classes[3]
#nn.ReLU = mx_classes[4]
#nn.ReLU6 = mx_classes[5]
#nn.Softmax = mx_classes[6]
#nn.AdaptiveAvgPool2d = mx_classes[7]

class AverageMeter(object):
  """Computes and stores the average and current value"""

  def __init__(self, name, fmt=':f'):
    self.name = name
    self.fmt = fmt
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

  def __str__(self):
    fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
    return fmtstr.format(**self.__dict__)

class ProgressMeter(object):

  def __init__(self, num_batches, meters, prefix=""):
    self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
    self.meters = meters
    self.prefix = prefix

  def display(self, batch):
    entries = [self.prefix + self.batch_fmtstr.format(batch)]
    entries += [str(meter) for meter in self.meters]
    print('\t'.join(entries))

  def _get_batch_fmtstr(self, num_batches):
    num_digits = len(str(num_batches // 1))
    fmt = '{:' + str(num_digits) + 'd}'
    return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def prepare_data_loaders(data_path,
                         train_batch_size=64,
                         val_batch_size=100,
                         workers=8):
  traindir = os.path.join(data_path, 'train')
  valdir = os.path.join(data_path, 'validation')
  normalize = transforms.Normalize(
      mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  train_dataset = datasets.ImageFolder(traindir,
                                       transforms.Compose([
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           normalize,
                                       ]))  # 1281167
  train_loader = torch.utils.data.DataLoader(
      train_dataset,
      batch_size=train_batch_size,
      shuffle=True,
      num_workers=workers,
      pin_memory=True)

  val_dataset = datasets.ImageFolder(valdir,
                                     transforms.Compose([
                                         transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         normalize,
                                     ]))  # len 50000

  val_loader = torch.utils.data.DataLoader(
      val_dataset,
      batch_size=val_batch_size,
      shuffle=False,
      num_workers=workers,
      pin_memory=True)  # len(500)

  return train_loader, val_loader

def accuracy(output, target, topk=(1,)):
  """Computes the accuracy over the k top predictions for the specified values of k"""
  with torch.no_grad():
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
      correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
      res.append(correct_k.mul_(100.0 / batch_size))
    return res

def validate(val_loader, model, criterion, device):
  batch_time = AverageMeter('Time', ':6.3f')
  losses = AverageMeter('Loss', ':.4e')
  top1 = AverageMeter('Acc@1', ':6.2f')
  top5 = AverageMeter('Acc@5', ':6.2f')
  progress = ProgressMeter(
      len(val_loader), [batch_time, losses, top1, top5], prefix='Test: ')

  # switch to evaluate mode
  model.eval()
  if not isinstance(model, nn.DataParallel):
    model = model.to(device)

  with torch.no_grad():
    end = time.time()
    for i, (images, target) in enumerate(val_loader):
      images = images.to(device, non_blocking=True)
      target = target.to(device, non_blocking=True)

      # compute output
      output = model(images)
      loss = criterion(output, target)

      # measure accuracy and record loss
      acc1, acc5 = accuracy(output, target, topk=(1, 5))
      losses.update(loss.item(), images.size(0))
      top1.update(acc1[0], images.size(0))
      top5.update(acc5[0], images.size(0))

      # measure elapsed time
      batch_time.update(time.time() - end)
      end = time.time()

      if i % 50 == 0:
        progress.display(i)

    # TODO: this should also be done with the ProgressMeter
    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(
        top1=top1, top5=top5))

  return top1.avg

def apply_mx_specs(model, mx_specs):
  for name, module in model.named_modules():
    if type(module) in mx_classes:
      if hasattr(module, 'apply_mx_specs'):
        module.apply_mx_specs(mx_specs)
      else:
        specs.mx_assert_test(mx_specs)
        module.mx_none = mx_specs is None
        module.mx_specs = specs.apply_mx_specs(mx_specs)
      print(f'Apply MX spec on {type(module)}')

name_to_model = {
    'resnet18': resnet18,
    'resnet50': resnet50,
    'mobilenet_v2': mobilenet_v2
}

if __name__ == '__main__':
  # Add config arguments
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_dir',
      default='/workspace/dataset/imagenet/pytorch',
      help='Data set directory.')
  parser.add_argument('--model', default=None, help='Model to use.')
  parser.add_argument(
      '--pretrained', default=None, help='Pretrained weights path.')
  parser.add_argument('--no_mx', action='store_true', help='Whether to do mx quantization.')
  parser.add_argument('--auto_module', action='store_true', help='Whether to use auto module.')
  parser.add_argument('--device', default='cuda')
  # Add MX arguments
  parser = add_mx_args(parser)
  args = parser.parse_args()
  print('Used arguments:', args)

  if not args.no_mx:
    # Process args to obtain mx_specs
    mx_specs = get_mx_specs(args)
    assert (mx_specs != None)

    mx_mapping.inject_pyt_ops(mx_specs)

  model = name_to_model[args.model]().to(args.device)
  model.load_state_dict(torch.load(args.pretrained))

  # Mainly used for quantization of python built-in ops like '+'.
  if args.auto_module:
    x = torch.randn([1, 3, 224, 224], dtype=torch.float32, device=args.device)
    from auto_module import wrap
    model = wrap(model, x)
  print(model)

  train_loader, val_loader = prepare_data_loaders(args.data_dir)
  criterion = nn.CrossEntropyLoss().to(args.device)
  validate(val_loader, model, criterion, args.device)
