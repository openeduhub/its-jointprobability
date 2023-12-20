from collections.abc import Collection
from pathlib import Path
from typing import Optional

import nlprep.spacy.props as nlp
import numpy as np
import pyro
import pyro.distributions as dist
import pyro.infer
import pyro.optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyro.nn.module import PyroModule, to_pyro_module_
from its_jointprobability.models.model import Model
from its_jointprobability.utils import (
    Quality_Result,
    device,
    quality_measures,
    texts_to_bow_tensor,
)
from icecream import ic


class ProdSLDA(Model):
    """
    A modification of the ProdLDA model to support supervized classification.
    """

    return_sites = ("label", "nu", "a")
    return_site_cat_dim = {"nu": 0, "a": -1, "label": -1}

    def __init__(
        self,
        voc_size: int,
        label_size: int,
        num_topics: int,
        layers: int,
        dropout: float,
        nu_loc: float = 0.0,
        nu_scale: float = 10.0,
        observe_negative_labels=torch.tensor(True),
    ):
        super().__init__()
        self.vocab_size = voc_size
        self.label_size = label_size
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
        self.nu_loc = nu_loc
        self.nu_scale = nu_scale
        self.observe_negative_labels = observe_negative_labels

    def model(
        self,
        docs: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        batch: Optional[Collection[int]] = None,
    ):
        num_full_data = docs.shape[0]

        if batch is not None:
            batch_size = len(batch)
            docs = docs[list(batch)]
            labels = labels[list(batch)] if labels is not None else None
        else:
            batch_size = num_full_data

        labels_plate = pyro.plate("labels", self.label_size, dim=-2)
        docs_plate = pyro.plate("documents", docs.shape[0], dim=-1)
        scale = pyro.poutine.scale(scale=num_full_data / batch_size)

        # # the label application coefficients
        with labels_plate:
            nu = pyro.sample(
                "nu",
                dist.Normal(
                    self.nu_loc * docs.new_ones(self.num_topics),
                    self.nu_scale * docs.new_ones(self.num_topics),
                ).to_event(1),
            )

        # pyro.module("decoder", self.decoder)
        with docs_plate, scale:
            # Dirichlet prior 𝑝(𝜃|𝛼) is replaced by a logistic-normal distribution
            logtheta_loc = docs.new_zeros(self.num_topics)
            logtheta_scale = docs.new_ones(self.num_topics)
            logtheta = pyro.sample(
                "logtheta", dist.Normal(logtheta_loc, logtheta_scale).to_event(1)
            )
            if len(logtheta.shape) > 2:
                logtheta = logtheta.squeeze(0)
            theta = F.softmax(logtheta, -1)

        # conditional distribution of 𝑤𝑛 is defined as
        # 𝑤𝑛|𝛽,𝜃 ~ Categorical(𝜎(𝛽𝜃))
        count_param = self.decoder(theta)

        with docs_plate, scale:
            # Currently, PyTorch Multinomial requires `total_count` to be homogeneous.
            # Because the numbers of words across documents can vary,
            # we will use the maximum count accross documents here.
            # This does not affect the result because Multinomial.log_prob does
            # not require `total_count` to evaluate the log probability.
            total_count = int(docs.sum(-1).max())
            pyro.sample(
                "obs",
                dist.Multinomial(total_count, count_param),
                obs=docs,
            )

            with labels_plate:
                a = pyro.sample("a", dist.Normal((nu.squeeze(-2) @ theta.T), 10))
                label = pyro.sample(
                    "label",
                    dist.Bernoulli(logits=a),  # type: ignore
                    obs=labels.T if labels is not None else None,
                    # obs_mask=torch.logical_or(
                    #     self.observe_negative_labels, labels.T.bool()
                    # )
                    # if labels is not None
                    # else None,
                    infer={"enumerate": "parallel"},
                )

    def logtheta_params(self, doc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logtheta_loc, logtheta_logvar = self.encoder(doc).split(self.num_topics, -1)
        logtheta_scale = (logtheta_logvar / 2).exp()

        return logtheta_loc, logtheta_scale

    def guide(
        self,
        docs: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        batch: Optional[Collection[int]] = None,
    ):
        num_full_data = docs.shape[0]

        if batch is not None:
            batch_size = len(batch)
            docs = docs[list(batch)]
            labels = labels[list(batch)] if labels is not None else None
        else:
            batch_size = num_full_data

        labels_plate = pyro.plate("labels", self.label_size, dim=-2)
        docs_plate = pyro.plate("documents", docs.shape[0], dim=-1)
        scale = pyro.poutine.scale(scale=num_full_data / batch_size)

        mu_q = pyro.param(
            "mu",
            lambda: torch.randn(self.label_size, self.num_topics)
            .unsqueeze(-2)
            .to(docs.device),
        )
        sigma_q = pyro.param(
            "sigma",
            lambda: docs.new_ones(self.label_size, self.num_topics).unsqueeze(-2),
            constraint=dist.constraints.positive,
        )

        with labels_plate:
            nu_q = pyro.sample("nu", dist.Normal(mu_q, sigma_q).to_event(1))

        with docs_plate, scale:
            logtheta_q = pyro.sample(
                "logtheta", dist.Normal(*self.logtheta_params(docs)).to_event(1)
            )
            theta_q = F.softmax(logtheta_q, -1)

            with labels_plate:
                a_q_scale = pyro.param(
                    "a_q_scale",
                    lambda: docs.new_ones([self.label_size, self.num_topics]),
                    constraint=dist.constraints.positive,
                )
                a_q = pyro.sample(
                    "a",
                    dist.Normal(
                        (nu_q.squeeze(-2) @ theta_q.T), (a_q_scale @ theta_q.T)
                    ),
                )

    def clean_up_posterior_samples(
        self, posterior_samples: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        if "label" in posterior_samples:
            posterior_samples["label"] = posterior_samples["label"].swapaxes(-1, -2)
        if "a" in posterior_samples:
            posterior_samples["a"] = posterior_samples["a"].swapaxes(-1, -2)
        if "nu" in posterior_samples:
            posterior_samples["nu"] = posterior_samples["nu"].squeeze(-2)

        return posterior_samples

    def beta(self):
        # beta matrix elements are the weights of the FC layer on the decoder
        weights = self.decoder.beta.weight.T
        return F.softmax(weights, dim=-1).detach()

    def draw_posterior_samples_from_texts(
        self,
        *texts: str,
        token_dict: dict[int, str],
        num_samples: int = 1000,
        return_sites: Optional[Collection[str]] = None,
    ):
        return_sites = return_sites or self.return_sites
        bow_tensor = texts_to_bow_tensor(*texts, token_dict=token_dict)
        return self.draw_posterior_samples(
            data_len=bow_tensor.shape[0],
            data_args=[bow_tensor],
            num_samples=num_samples,
            return_sites=return_sites,
        )


def import_data(
    path: Path,
) -> tuple[ProdSLDA, dict[int, str], list[str], torch.Tensor, torch.Tensor]:
    args = torch.load(path / "prodslda_args")
    kwargs = torch.load(path / "prodslda_kwargs")
    prodslda = ProdSLDA(*args, **kwargs).to(device)
    state_dict = torch.load(path / "prodslda_state_dict", map_location=device)
    prodslda.load_state_dict(state_dict)
    pyro.get_param_store().load(path / "pyro_store", map_location=device)
    dictionary = torch.load(path / "dictionary")
    labels = torch.load(path / "labels")

    train_data = torch.load(path / "train_data", map_location=device)
    train_labels = torch.load(path / "train_labels", map_location=device)

    return prodslda, dictionary, labels, train_data, train_labels


def retrain_model(path: Path, n=None) -> ProdSLDA:
    train_data: torch.Tensor = torch.load(path / "train_data", map_location=device)
    train_labels: torch.Tensor = torch.load(path / "train_labels", map_location=device)

    if n is not None:
        train_data = train_data[:n]
        train_labels = train_labels[:n]

    pyro.get_param_store().clear()

    prodslda = ProdSLDA(
        voc_size=train_data.shape[-1],
        label_size=train_labels.shape[-1],
        num_topics=400,
        layers=200,
        dropout=0.2,
        nu_loc=0.0,
        nu_scale=10.0,
        observe_negative_labels=torch.tensor(True, device=device),
    ).to(device)

    prodslda.run_svi(
        train_args=[train_data, train_labels],
        train_data_len=train_data.shape[0],
        elbo=pyro.infer.TraceGraph_ELBO(num_particles=3),
        max_epochs=1000,
        batch_size=n,
    )

    prodslda = prodslda.eval()
    torch.save(prodslda, path / "prodslda")

    return prodslda
