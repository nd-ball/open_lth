# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
from functools import partial

from foundations.hparams import PruningHparams
from pruning import sparse_global, random_global

registered_strategies = {'sparse_global': sparse_global.Strategy, 'random_global': random_global.Strategy}


def get(pruning_hparams: PruningHparams):
    """Get the pruning function."""

    return partial(registered_strategies[pruning_hparams.pruning_strategy].prune,
                   copy.deepcopy(pruning_hparams))


def get_pruning_hparams(pruning_strategy: str) -> type:
    """Get a complete lottery schema as specialized for a particular pruning strategy."""

    if pruning_strategy not in registered_strategies:
        raise ValueError('Pruning strategy {} not found.'.format(pruning_strategy))

    return registered_strategies[pruning_strategy].get_pruning_hparams()
