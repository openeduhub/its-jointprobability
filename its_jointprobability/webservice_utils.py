from collections.abc import Collection, Iterable, Sequence
from pathlib import Path

import nlprep.spacy.props as nlp
import numpy as np
import pyro
import pyro.distributions as dist
import pyro.infer
import pyro.optim
import torch
import torch.nn as nn
import torch.nn.functional as F

from its_jointprobability.models.model import Model
from its_jointprobability.models.prodslda import ProdSLDA
from its_jointprobability.utils import (
    device,
    get_random_batch_strategy,
    texts_to_bow_tensor,
    labels_to_tensor,
)


def update_with_new_data(
    model: ProdSLDA,
    texts: Collection[str],
    labels: Collection[Collection[str]],
    train_data: torch.Tensor,
    train_labels: torch.Tensor,
    token_dict: dict[int, str],
    label_values: Sequence[str],
    **kwargs,
):
    model.train()
    bows_tensor = texts_to_bow_tensor(*texts, token_dict=token_dict)
    labels_tensor = labels_to_tensor(*labels, label_values=label_values)
    n = bows_tensor.shape[0]

    def get_random_batch_strategy_modified(length, batch_size):
        random_batch_strategy = get_random_batch_strategy(length, batch_size)
        while True:
            final_batch, batch = random_batch_strategy.__next__()
            # always include the new data
            batch = list(set(batch) | set(range(n)))
            yield final_batch, batch

    model.run_svi(
        train_data_len=bows_tensor.shape[0] + train_data.shape[0],
        train_args=[
            torch.cat([bows_tensor, train_data]),
            torch.cat([labels_tensor, train_labels]),
        ],
        elbo=pyro.infer.TraceEnum_ELBO(num_particles=3),
        batch_strategy_factory=get_random_batch_strategy_modified,
        min_epochs=10,
        max_epochs=100,
        min_z_score=1,
        z_score_num=5,
        **kwargs,
    )
