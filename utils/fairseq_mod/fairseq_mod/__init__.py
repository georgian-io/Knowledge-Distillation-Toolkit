# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

__all__ = ['pdb']
__version__ = '0.9.0'

import sys

# backwards compatibility to support `from fairseq.meters import AverageMeter`
from fairseq_mod.logging import meters, metrics, progress_bar  # noqa
sys.modules['fairseq.meters'] = meters
sys.modules['fairseq.metrics'] = metrics
sys.modules['fairseq.progress_bar'] = progress_bar

import fairseq_mod.criterions  # noqa
import fairseq_mod.models  # noqa
import fairseq_mod.modules  # noqa
import fairseq_mod.optim  # noqa
import fairseq_mod.optim.lr_scheduler  # noqa
import fairseq_mod.pdb  # noqa
import fairseq_mod.scoring  # noqa
import fairseq_mod.tasks  # noqa
import fairseq_mod.token_generation_constraints  # noqa

import fairseq_mod.benchmark  # noqa
import fairseq_mod.model_parallel  # noqa
