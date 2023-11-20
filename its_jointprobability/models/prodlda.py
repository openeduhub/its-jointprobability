from collections.abc import Collection
from typing import Optional

import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyro.nn.module import to_pyro_module_
from its_jointprobability.models.model import Model


class ProdLDA(Model):
    return_sites = ("logtheta",)
    return_site_cat_dim = {"logtheta": -2}

    def __init__(self, voc_size: int, num_topics: int, layers: int, dropout: float):
        super().__init__()
        self.vocab_size = voc_size
        self.num_topics = num_topics
        self.decoder = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_topics, voc_size),
            nn.BatchNorm1d(voc_size, affine=False),
            nn.Softmax(-1),
        )
        to_pyro_module_(self.decoder)
        self.encoder = nn.Sequential(
            nn.Linear(voc_size, layers),
            nn.Softplus(),
            nn.Linear(layers, layers),
            nn.Softplus(),
            nn.Dropout(dropout),
            nn.Linear(layers, num_topics * 2),
            nn.BatchNorm1d(num_topics * 2, affine=False),
        )
        to_pyro_module_(self.encoder)

    def logtheta_params(self, doc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logtheta_loc, logtheta_logvar = self.encoder(doc).split(self.num_topics, -1)
        logtheta_scale = (logtheta_logvar / 2).exp()

        return logtheta_loc, logtheta_scale

    def model(self, docs: torch.Tensor, batch: Optional[Collection[int]] = None):
        num_full_data = docs.shape[0]

        if batch is not None:
            batch_size = len(batch)
            docs = docs[list(batch)]
        else:
            batch_size = num_full_data

        docs_plate = pyro.plate("documents", docs.shape[0], dim=-1)
        scale = pyro.poutine.scale(scale=num_full_data / batch_size)

        with docs_plate, scale:
            logtheta_loc = docs.new_zeros(self.num_topics)
            logtheta_scale = docs.new_ones(self.num_topics)
            logtheta = pyro.sample(
                "logtheta", dist.Normal(logtheta_loc, logtheta_scale).to_event(1)
            )
            theta = F.softmax(logtheta, -1)

        count_param = self.decoder(theta)

        with docs_plate, scale:
            total_count = int(docs.sum(-1).max())
            pyro.sample(
                "obs",
                dist.Multinomial(total_count, count_param),
                obs=docs,
            )

    def guide(
        self, docs: torch.Tensor, batch: Optional[Collection[int]] = None
    ) -> torch.Tensor:
        num_full_data = docs.shape[0]

        if batch is not None:
            batch_size = len(batch)
            docs = docs[list(batch)]
        else:
            batch_size = num_full_data

        docs_plate = pyro.plate("documents", docs.shape[0], dim=-1)
        scale = pyro.poutine.scale(scale=num_full_data / batch_size)

        with docs_plate, scale:
            return pyro.sample(
                "logtheta", dist.Normal(*self.logtheta_params(docs)).to_event(1)
            )

    def beta(self):
        # beta matrix elements are the weights of the FC layer on the decoder
        weights = self.decoder.beta.weight.T
        return F.softmax(weights, dim=-1).detach()

    def clean_up_posterior_samples(
        self, posterior_samples: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        if "label" in posterior_samples:
            posterior_samples["label"] = posterior_samples["label"].swapaxes(-1, -2)

        return posterior_samples
