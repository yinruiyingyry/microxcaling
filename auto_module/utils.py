from torch import nn
import traceback
import functools
from contextlib import contextmanager

OP_COUNTS: str = "op_counts"

class Context:
  def __init__(self) -> None:
    self._enable_replacement = False

  def enable_replacement(self):
    self._enable_replacement = True

  def disable_replacement(self):
    self._enable_replacement = False

  def is_replacement_enabled(self):
    return self._enable_replacement

_context = Context()

@contextmanager
def enable_replacement():
  original_value = _context.is_replacement_enabled()
  try:
    _context.enable_replacement()
    yield
  finally:
    if original_value:
      _context.enable_replacement()
    else:
      _context.disable_replacement()


@contextmanager
def disable_replacement():
  original_value = _context.is_replacement_enabled()
  try:
    _context.disable_replacement()
    yield
  finally:
    if original_value:
      _context.enable_replacement()
    else:
      _context.disable_replacement()


def get_module_or_context() -> nn.Module:
  for frame in traceback.walk_stack(None):
    f_locals = frame[0].f_locals
    if "self" in f_locals and isinstance(f_locals["self"], nn.Module):
      return f_locals["self"]
  return _context


def get_op_id(op_name: str) -> str:
  module = get_module_or_context()
  if not hasattr(module, OP_COUNTS):
    setattr(module, OP_COUNTS, {})
  if op_name in getattr(module, OP_COUNTS):
    getattr(module, OP_COUNTS)[op_name] += 1
  else:
    getattr(module, OP_COUNTS)[op_name] = 1
  op_count = getattr(module, OP_COUNTS)[op_name]
  op_id = op_name + "_" + str(op_count)
  return module, op_id


def wrap_stateless_function(python_module, func_name: str, replace_module: nn.Module) -> None:
  module_name = python_module.__name__ \
    if "." not in python_module.__name__ \
    else python_module.__name__.split(".")[-1]
  op_name = "wrapped_" + module_name + "_" + func_name
  original_func = getattr(python_module, func_name)

  @functools.wraps(original_func)
  def func(*args, **kwargs):
    if not _context.is_replacement_enabled():
      return original_func(*args, **kwargs)
    with disable_replacement():
      module, op_id = get_op_id(op_name=op_name)
      if isinstance(module, replace_module):
        return original_func(*args, **kwargs)
      if not hasattr(module, op_id):
        setattr(module, op_id, replace_module())

      return getattr(module, op_id)(*args, **kwargs)

  setattr(python_module, func_name, func)
