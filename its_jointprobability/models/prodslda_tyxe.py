from collections.abc import Collection
from pathlib import Path
from typing import Optional
from pyro import poutine

import pyro.distributions as dist
import pyro
import pyro.optim
import pyro.infer
import pyro.infer.autoguide
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyxe
from its_jointprobability.models.model import Model
from its_jointprobability.utils import device, texts_to_bow_tensor, quality_measures

from icecream import ic


def get_bayes_encoder(
    in_dim: int, out_dim: int, hid_dim: int, dropout: float
) -> tyxe.VariationalBNN:
    encoder = nn.Sequential(
        nn.Linear(in_dim, hid_dim),
        nn.Softplus(),
        nn.Linear(hid_dim, hid_dim),
        nn.Softplus(),
        nn.Dropout(dropout),
        nn.Linear(hid_dim, out_dim * 2),
        nn.BatchNorm1d(out_dim * 2, affine=False),
    ).to(device)
    prior = tyxe.priors.IIDPrior(dist.Normal(0, 1))
    likelihood = tyxe.likelihoods.HomoskedasticGaussian(out_dim, scale=1)
    guide = tyxe.guides.AutoNormal

    return tyxe.VariationalBNN(encoder, prior, likelihood, guide)


def get_bayes_decoder(in_dim: int, out_dim: int, dropout: float) -> tyxe.VariationalBNN:
    decoder = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim, affine=False),
        nn.Softmax(-1),
    ).to(device)
    prior = tyxe.priors.IIDPrior(dist.Normal(0, 1))
    likelihood = tyxe.likelihoods.HomoskedasticGaussian(out_dim, scale=1)
    guide = tyxe.guides.AutoNormal

    return tyxe.VariationalBNN(decoder, prior, likelihood, guide)


class ProdSLDA_tyxe(Model):
    return_sites = ("label", "nu", "a")
    return_site_cat_dim = {"nu": 0, "a": -1, "label": -1}

    def __init__(
        self,
        voc_size: int,
        label_size: int,
        num_topics: int,
        layers: int,
        dropout: float = 0.2,
        nu_loc: float = 0.0,
        nu_scale: float = 10.0,
        observe_negative_labels=torch.tensor(True),
    ):
        super().__init__("ProdSLDA_tyxe")
        self.vocab_size = voc_size
        self.label_size = label_size
        self.num_topics = num_topics
        self.encoder = get_bayes_encoder(
            in_dim=voc_size, out_dim=num_topics, hid_dim=layers, dropout=dropout
        )
        self.decoder = get_bayes_decoder(
            in_dim=num_topics, out_dim=voc_size, dropout=dropout
        )
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

        # logtheta_loc, logtheta_scale = self.encoder(docs)
        logtheta_loc, logtheta_logvar = self.encoder(docs).split(self.num_topics, -1)
        logtheta_scale = (0.5 * logtheta_logvar).exp()  # Enforces positivity

        with docs_plate, scale:
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
                    obs_mask=torch.logical_or(
                        self.observe_negative_labels, labels.T.bool()
                    )
                    if labels is not None
                    else None,
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

        # logtheta_loc, logtheta_scale = self.encoder(docs)
        logtheta_loc, logtheta_logvar = self.encoder.guided_forward(docs).split(
            self.num_topics, -1
        )
        logtheta_scale = (0.5 * logtheta_logvar).exp()  # Enforces positivity

        with docs_plate, scale:
            # Dirichlet prior 𝑝(𝜃|𝛼) is replaced by a logistic-normal distribution,
            # where μ and Σ are the encoder network outputs
            logtheta_q = pyro.sample(
                "logtheta", dist.Normal(logtheta_loc, logtheta_scale).to_event(1)
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


def retrain_model(path: Path) -> ProdSLDA_tyxe:
    train_data: torch.Tensor = torch.load(path / "train_data", map_location=device)
    train_labels: torch.Tensor = torch.load(path / "train_labels", map_location=device)

    pyro.get_param_store().clear()

    torch.set_default_device(device)

    pyro.set_rng_seed(0)
    prodslda = ProdSLDA_tyxe(
        voc_size=train_data.shape[-1],
        label_size=train_labels.shape[-1],
        num_topics=100,
        layers=100,
        dropout=0.2,
        nu_loc=0.0,
        nu_scale=5.0,
        observe_negative_labels=torch.tensor(True, device=device),
    ).to(device)

    # with tyxe.poutine.local_reparameterization():
    prodslda.run_svi(
        train_args=[train_data, train_labels],
        train_data_len=train_data.shape[0],
        elbo=pyro.infer.TraceEnum_ELBO(num_particles=1),
        initial_lr=0.01,
        # batch_size=1000,
    )

    # torch.save(prodslda, path / "prodslda_tyxe")
    # pyro.get_param_store().save(path / "pyro_store_tyxe")

    return prodslda


def train_here_qualities() -> ProdSLDA_tyxe:
    ic.disable()
    path = Path.cwd() / "data"
    prodslda = retrain_model(path)

    train_data: torch.Tensor = torch.load(path / "train_data", map_location=device)
    train_labels: torch.Tensor = torch.load(path / "train_labels", map_location=device)

    samples = prodslda.draw_posterior_samples(
        train_data.shape[-2], data_args=[train_data]
    )["label"]
    print(quality_measures(samples, train_labels, mean_dim=0, cutoff=None))
    return prodslda
