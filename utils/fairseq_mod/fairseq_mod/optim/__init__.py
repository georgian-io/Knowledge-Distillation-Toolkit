# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import os
from argparse import Namespace
from typing import Union

from fairseq_mod import registry
from fairseq_mod.optim.bmuf import FairseqBMUF  # noqa
from fairseq_mod.optim.fairseq_optimizer import (  # noqa
    FairseqOptimizer,
    LegacyFairseqOptimizer,
)
from fairseq_mod.optim.fp16_optimizer import FP16Optimizer, MemoryEfficientFP16Optimizer
from fairseq_mod.optim.shard import shard_
from omegaconf import DictConfig


__all__ = [
    "FairseqOptimizer",
    "FP16Optimizer",
    "MemoryEfficientFP16Optimizer",
    "shard_",
]


(
    _build_optimizer,
    register_optimizer,
    OPTIMIZER_REGISTRY,
    OPTIMIZER_DATACLASS_REGISTRY,
) = registry.setup_registry("--optimizer", base_class=FairseqOptimizer, required=True)


def build_optimizer(
    optimizer_cfg: Union[DictConfig, Namespace], params, *extra_args, **extra_kwargs
):
    if all(isinstance(p, dict) for p in params):
        params = [t for p in params for t in p.values()]
    params = list(filter(lambda p: p.requires_grad, params))
    return _build_optimizer(optimizer_cfg, params, *extra_args, **extra_kwargs)


# automatically import any Python files in the optim/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith(".py") and not file.startswith("_"):
        file_name = file[: file.find(".py")]
        importlib.import_module("fairseq_mod.optim." + file_name)
