import torch
from torch import nn

class Add(nn.Module):
  def __init__(self) -> None:
    super().__init__()

  def forward(self, a, b):
    return a + b

class Mul(nn.Module):
  def __init__(self) -> None:
    super().__init__()

  def forward(self, a, b):
    return a * b

class Div(nn.Module):
  def __init__(self) -> None:
    super().__init__()

  def forward(self, a, b):
    return a / b

class Rdiv(nn.Module):
  def __init__(self) -> None:
    super().__init__()

  def forward(self, a, b):
    return b / a

class Sub(nn.Module):
  def __init__(self) -> None:
    super().__init__()

  def forward(self, a, b):
    return a - b

class Rsub(nn.Module):
  def __init__(self) -> None:
    super().__init__()

  def forward(self, a, b):
    return b - a

class LessThan(nn.Module):
  def __init__(self) -> None:
    super().__init__()

  def forward(self, a, b):
    return a < b

class To(nn.Module):
  def __init__(self) -> None:
    super().__init__()

  def forward(self, tensor, *args, **kwargs):
    return tensor.to(*args, **kwargs)

class Type(nn.Module):
  def __init__(self) -> None:
    super().__init__()

  def forward(self, tensor, *args, **kwargs):
    return tensor.type(*args, **kwargs)

class Where(nn.Module):
  def __init__(self) -> None:
    super().__init__()

  def forward(self, condition, input, other, *, out=None):
    return torch.where(condition, input, other, out=out)

class TorchAdd(nn.Module):
  def __init__(self) -> None:
    super().__init__()

  def forward(self, input, other, *, alpha=1, out=None):
    return torch.add(input, other, alpha=alpha, out=out)

class TorchMul(nn.Module):
  def __init__(self) -> None:
    super().__init__()

  def forward(self, input, other, *, out=None):
    return torch.mul(input, other, out=out)

class TorchMatMul(nn.Module):
  def __init__(self) -> None:
    super().__init__()

  def forward(self, input, other, *, out=None):
    return torch.matmul(input, other, out=out)

class TorchSqrt(nn.Module):
  def __init__(self) -> None:
    super().__init__()

  def forward(self, input, *, out=None):
    return torch.sqrt(input, out=out)
