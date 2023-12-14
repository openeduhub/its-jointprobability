from __future__ import annotations

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
from icecream import ic
from its_jointprobability.models.model import Model, eval_model
from its_jointprobability.utils import device, texts_to_bow_tensor
from pyro.infer.enum import partial


class ProdSLDA(Model):
    """
    A modification of the ProdLDA model to support supervised classification.
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
        encoder=None,
        decoder=None,
    ):
        super().__init__()
        self.voc_size = voc_size
        self.label_size = label_size
        self.num_topics = num_topics
        self.layers = layers
        self.dropout = dropout
        self.nu_loc = nu_loc
        self.nu_scale = nu_scale
        self.observe_negative_labels = observe_negative_labels

        if decoder is None:
            self.decoder = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(num_topics, voc_size),
                nn.BatchNorm1d(voc_size, affine=False),
                nn.Softmax(-1),
            )
        else:
            self.decoder = decoder

        if encoder is None:
            self.encoder = nn.Sequential(
                nn.Linear(voc_size, layers),
                nn.Softplus(),
                nn.Linear(layers, layers),
                nn.Softplus(),
                nn.Dropout(dropout),
                nn.Linear(layers, num_topics * 2),
                nn.BatchNorm1d(num_topics * 2, affine=False),
            )
        else:
            self.encoder = encoder

        self.device = device

        # to_pyro_module_(self)

    def bayesian_update(
        self, docs: torch.Tensor, labels: torch.Tensor, num_particles=3, *args, **kwargs
    ) -> tuple[ProdSLDA, list[float]]:
        elbo = pyro.infer.Trace_ELBO(num_particles=num_particles)

        # run svi on the given data
        param_store = pyro.get_param_store()
        with param_store.scope() as svi:
            losses = self.run_svi(
                elbo=elbo,
                train_data_len=docs.shape[-2],
                train_args=[docs, labels],
                *args,
                **kwargs,
            )

        self.eval()
        ic(self.encoder.training)
        ic(self.decoder.training)
        ic(self.encoder.state_dict())

        with param_store.scope(svi):
            nu_loc = pyro.param("nu_loc").detach()
            nu_scale = pyro.param("nu_scale").detach()
            a_scale = pyro.param("a_scale").detach()

        new_model = ProdSLDA(
            voc_size=self.voc_size,
            label_size=self.label_size,
            num_topics=self.num_topics,
            layers=self.layers,
            dropout=self.dropout,
            observe_negative_labels=self.observe_negative_labels,
        )

        new_model.logtheta_prior = self.logtheta_posterior
        new_model.nu_prior = partial(  # type: ignore
            self.nu_posterior,
            nu_loc=nu_loc,
            nu_scale=nu_scale,
        )
        new_model.a_prior = partial(  # type: ignore
            self.a_posterior,
            a_scale=a_scale,
        )

        ic(new_model.encoder.training)
        ic(new_model.decoder.training)
        ic(new_model.encoder.state_dict())

        # no longer use the posterior distribution in the predictive
        new_model.predictive = new_model.model_predictive

        return new_model, losses

    def model_predictive(self, *args, **kwargs) -> pyro.infer.Predictive:
        """
        Only draw samples from the model (not the guide).

        We use this for the Bayesian updates, as they set the model's priors
        to the approximated posteriors and reset the guides.
        Thus, these guides are no longer useful for inference.
        """
        return pyro.infer.Predictive(model=self.model, *args, **kwargs)

    def logtheta_prior(
        self,
        docs: torch.Tensor,
        *args,
        logtheta_loc: float = 0.0,
        logtheta_scale: float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        """The prior on the per-document assignment of topics."""
        return pyro.sample(
            "logtheta",
            dist.Normal(
                logtheta_loc * docs.new_ones(self.num_topics),
                logtheta_scale * docs.new_ones(self.num_topics),
            ).to_event(1),
        )

    def logtheta_posterior(self, docs, *args, **kwargs) -> torch.Tensor:
        logtheta_loc, logtheta_logvar = self.encoder(docs).split(self.num_topics, -1)
        logtheta_scale = (logtheta_logvar / 2).exp()

        return pyro.sample(
            "logtheta", dist.Normal(logtheta_loc, logtheta_scale).to_event(1)
        )

    def a_prior(
        self, nu, theta, *args, a_scale: float = 10.0, **kwargs
    ) -> torch.Tensor:
        """The prior on the applicability of each label, given the topic mix."""
        return pyro.sample(
            "a",
            dist.Normal(
                torch.matmul(nu, theta.swapaxes(-1, -2)).squeeze(-2),
                a_scale,
            ),
        )

    def a_posterior(
        self, nu, theta, *args, a_scale: Optional[torch.Tensor] = None, **kwargs
    ):
        """The posterior on the applicability of each label, given the topic mix."""

        a_scale = (
            a_scale
            if a_scale is not None
            else pyro.param(
                "a_scale",
                lambda: torch.ones(
                    [self.label_size, 1, self.num_topics], device=self.device
                ),
                constraint=dist.constraints.positive,
            )
        )
        return pyro.sample(
            "a",
            dist.Normal(
                torch.matmul(nu, theta.swapaxes(-1, -2)).squeeze(-2),
                torch.matmul(a_scale, theta.swapaxes(-1, -2)).squeeze(-2),
            ),
        )

    def nu_prior(
        self, *args, nu_loc: float = 0.0, nu_scale: float = 10.0, **kwargs
    ) -> torch.Tensor:
        """The prior on the label x topic relevance-matrix."""
        return pyro.sample(
            "nu",
            dist.Normal(
                nu_loc
                * torch.ones(self.label_size, 1, self.num_topics, device=self.device),
                nu_scale
                * torch.ones(self.label_size, 1, self.num_topics, device=self.device),
            ).to_event(1),
        )

    def nu_posterior(
        self,
        *args,
        nu_loc: Optional[torch.Tensor] = None,
        nu_scale: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """The posterior on the label x topic revelance-matrix."""
        nu_loc = (
            nu_loc
            if nu_loc is not None
            else pyro.param(
                "nu_loc",
                lambda: torch.randn(
                    self.label_size, 1, self.num_topics, device=self.device
                ),
            )
        )
        nu_scale = (
            nu_scale
            if nu_scale is not None
            else pyro.param(
                "nu_scale",
                lambda: torch.ones(
                    self.label_size, 1, self.num_topics, device=self.device
                ),
                constraint=dist.constraints.positive,
            )
        )
        return pyro.sample("nu", dist.Normal(nu_loc, nu_scale).to_event(1))

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

        with labels_plate:
            nu = self.nu_prior(docs=docs)
            # ic(nu.shape)

        with docs_plate, scale:
            logtheta = self.logtheta_prior(docs=docs)
            if len(logtheta.shape) > 2:
                logtheta = logtheta.squeeze(0)
            theta = F.softmax(logtheta, -1)

            pyro.module("encoder", self.encoder)
            count_param = self.decoder(theta)

            total_count = int(docs.sum(-1).max())
            pyro.sample(
                "obs",
                dist.Multinomial(total_count, count_param),
                obs=docs,
            )

            with labels_plate:
                a = self.a_prior(docs=docs, nu=nu, theta=theta)
                # ic(a.shape)
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

        with labels_plate:
            nu_q = self.nu_posterior(docs=docs)
            # ic(nu_q.shape)

        with docs_plate, scale:
            pyro.module("decoder", self.decoder)
            logtheta_q = self.logtheta_posterior(docs=docs)
            theta_q = F.softmax(logtheta_q, -1)
            # ic(theta_q.shape)

            with labels_plate:
                a_q = self.a_posterior(docs=docs, nu=nu_q, theta=theta_q)
                # ic(a_q.shape)

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


def retrain_model(path: Path, n=None, seed=0) -> ProdSLDA:
    train_data: torch.Tensor = torch.load(path / "train_data", map_location=device)[:n]
    train_labels: torch.Tensor = torch.load(path / "train_labels", map_location=device)[
        :n
    ]
    labels: torch.Tensor = torch.load(path / "labels")

    pyro.get_param_store().clear()

    pyro.set_rng_seed(seed)

    prodslda = ProdSLDA(
        voc_size=train_data.shape[-1],
        label_size=train_labels.shape[-1],
        num_topics=100,
        layers=50,
        dropout=0.2,
        nu_loc=0.0,
        nu_scale=10.0,
        observe_negative_labels=torch.tensor(True, device=device),
    ).to(device)

    prodslda.run_svi(
        train_args=[train_data, train_labels],
        train_data_len=train_data.shape[0],
        elbo=pyro.infer.TraceGraph_ELBO(num_particles=3),
        max_epochs=250,
        batch_size=n,
    )

    prodslda.encoder.eval()
    prodslda.decoder.eval()
    # torch.save(prodslda, path / "prodslda")

    eval_model(prodslda, train_data, train_labels, labels)

    return prodslda


def retrain_model_bayes(path: Path, n=None, seed=0) -> ProdSLDA:
    train_data: torch.Tensor = torch.load(path / "train_data", map_location=device)[:n]
    train_labels: torch.Tensor = torch.load(path / "train_labels", map_location=device)[
        :n
    ]
    labels: torch.Tensor = torch.load(path / "labels")

    pyro.get_param_store().clear()
    pyro.set_rng_seed(seed)

    prodslda = ProdSLDA(
        voc_size=train_data.shape[-1],
        label_size=train_labels.shape[-1],
        num_topics=100,
        layers=50,
        dropout=0.2,
        nu_loc=0.0,
        nu_scale=10.0,
        observe_negative_labels=torch.tensor(True, device=device),
    ).to(device)

    new_prodslda, losses = prodslda.bayesian_update(
        train_data[:200], train_labels[:200], max_epochs=250
    )

    print("evaluation of first model")
    eval_model(new_prodslda, train_data[:200], train_labels[:200], labels)

    new_prodslda, losses = new_prodslda.bayesian_update(
        train_data[200:400], train_labels[200:400], max_epochs=250
    )

    new_prodslda = new_prodslda.eval()
    # torch.save(prodslda, path / "prodslda")

    print("evaluation of second model")
    print("initial dataset")
    eval_model(new_prodslda, train_data[:200], train_labels[:200], labels)
    print("new dataset")
    eval_model(new_prodslda, train_data[200:400], train_labels[200:400], labels)
    print("whole dataset")
    eval_model(new_prodslda, train_data[:400], train_labels[:400], labels)

    return new_prodslda
