from collections.abc import Collection
from pathlib import Path
from typing import Optional

import nlprep.spacy.props as nlp
import numpy as np
import pyro
import pyro.distributions as dist
import pyro.infer
import pyro.infer.autoguide
import pyro.optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyro.nn.module import PyroModule
from its_jointprobability.models.model import Model
from its_jointprobability.utils import device, texts_to_bow_tensor
from icecream import ic


class SLDA(Model):
    return_sites = ("label", "a")
    return_site_cat_dim = {"a": -2, "label": -2}

    def __init__(
        self,
        voc_size: int,
        label_size: int,
        num_topics: int,
        beta_loc: float = 0.0,
        beta_scale: float = 1.0,
        nu_loc: float = 0.0,
        nu_scale: float = 1.0,
        observe_negative_labels=torch.tensor(True),
    ):
        super().__init__()
        self.vocab_size = voc_size
        self.label_size = label_size
        self.num_topics = num_topics
        self.nu_loc = nu_loc
        self.nu_scale = nu_scale
        self.beta_loc = beta_loc
        self.beta_scale = beta_scale
        self.observe_negative_labels = observe_negative_labels

    def model(
        self,
        docs: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        batch: Optional[Collection[int]] = None,
    ):
        docs = docs.unsqueeze(-3)
        labels = labels.swapaxes(-1, -2).unsqueeze(-2) if labels is not None else None

        docs_plate = pyro.plate(
            "documents-plate", docs.shape[-2], dim=-1, subsample=batch
        )
        topics_plate = pyro.plate("topics-plate", self.num_topics, dim=-2)
        labels_plate = pyro.plate("labels-plate", self.label_size, dim=-3)

        with topics_plate:
            with labels_plate:
                # the label application coefficients
                nu = pyro.sample("nu", dist.Normal(docs.new_zeros(1), docs.new_ones(1)))

            # the word application coefficients
            beta = pyro.sample(
                "beta",
                dist.Normal(docs.new_zeros(1), docs.new_ones(1))
                .expand(torch.Size([1, self.num_topics, 1, self.vocab_size]))
                .to_event(1),
            )

        with docs_plate as ind:
            with topics_plate:
                # Dirichlet prior 𝑝(𝜃|𝛼) is replaced by a logistic-normal distribution
                logtheta_loc = docs.new_zeros(1)
                logtheta_scale = docs.new_ones(1)
                logtheta = pyro.sample(
                    "logtheta",
                    dist.Normal(logtheta_loc, logtheta_scale).expand(
                        torch.Size([1, self.num_topics, len(ind)])
                    ),
                )
                theta = F.softmax(logtheta, -2)

            count_param = pyro.sample(
                "count_param",
                dist.Normal(
                    torch.matmul(
                        theta.unsqueeze(-1).swapaxes(-1, -3), beta.swapaxes(-2, -3)
                    ),
                    10,
                ).to_event(1),
            )

            total_count = int(docs[..., ind, :].sum(-1).max())
            pyro.sample(
                "obs",
                dist.Multinomial(total_count, logits=count_param),
                obs=docs[..., ind, :],
            )

            with labels_plate:
                a = pyro.sample(
                    "a",
                    dist.Normal(torch.matmul(nu.swapaxes(-1, -2), theta), 10),
                )
                label = pyro.sample(
                    "label",
                    dist.Bernoulli(logits=a),  # type: ignore
                    obs=labels[..., ind] if labels is not None else None,
                    obs_mask=torch.logical_or(
                        self.observe_negative_labels, labels[..., ind].bool()
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
        docs = docs.unsqueeze(-3)
        labels = labels.swapaxes(-1, -2).unsqueeze(-2) if labels is not None else None

        docs_plate = pyro.plate(
            "documents-plate", docs.shape[-2], dim=-1, subsample=batch
        )
        topics_plate = pyro.plate("topics-plate", self.num_topics, dim=-2)
        labels_plate = pyro.plate("labels-plate", self.label_size, dim=-3)

        nu_loc = pyro.param(
            "nu_loc", lambda: docs.new_zeros(self.label_size, self.num_topics, 1)
        )
        nu_scale = pyro.param(
            "nu_scale",
            lambda: docs.new_ones(self.label_size, self.num_topics, 1),
            constraint=dist.constraints.positive,
        )

        beta_loc = pyro.param(
            "beta_loc", lambda: docs.new_zeros(1, self.num_topics, 1, self.vocab_size)
        )
        beta_scale = pyro.param(
            "beta_scale",
            lambda: docs.new_ones(1, self.num_topics, 1, self.vocab_size),
            constraint=dist.constraints.positive,
        )

        logtheta_loc = pyro.param(
            "logtheta_loc", lambda: docs.new_zeros(1, self.num_topics, 1)
        )
        logtheta_scale = pyro.param(
            "logtheta_scale",
            lambda: docs.new_ones(1, self.num_topics, 1),
            constraint=dist.constraints.positive,
        )

        a_scale = pyro.param(
            "a_scale",
            lambda: docs.new_ones(self.label_size, 1, 1),
            constraint=dist.constraints.positive,
        )

        with topics_plate:
            with labels_plate:
                # the label application coefficients
                nu = pyro.sample("nu", dist.Normal(nu_loc, nu_scale))

            # the word application coefficients
            beta = pyro.sample("beta", dist.Normal(beta_loc, beta_scale).to_event(1))

        # pyro.module("decoder", self.decoder)
        with docs_plate as ind:
            with topics_plate:
                # Dirichlet prior 𝑝(𝜃|𝛼) is replaced by a logistic-normal distribution
                logtheta = pyro.sample(
                    "logtheta", dist.Normal(logtheta_loc, logtheta_scale)
                )
                theta = F.softmax(logtheta, -2)

            count_param = pyro.sample(
                "count_param",
                dist.Normal(
                    torch.matmul(
                        theta.unsqueeze(-1).swapaxes(-1, -3), beta.swapaxes(-2, -3)
                    ),
                    10,
                ).to_event(1),
            )

            with labels_plate:
                a_loc = torch.matmul(nu.swapaxes(-1, -2), theta)
                a = pyro.sample("a", dist.Normal(a_loc, a_scale))

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


def import_data(
    path: Path,
) -> tuple[SLDA, dict[int, str], list[str], torch.Tensor, torch.Tensor]:
    import gc

    torch.set_default_device(device)

    args = torch.load(path / "prodslda_args")
    kwargs = torch.load(path / "prodslda_kwargs")
    slda = SLDA(*args, **kwargs).to(device)
    state_dict = torch.load(path / "prodslda_state_dict", map_location=device)
    slda.load_state_dict(state_dict)
    pyro.get_param_store().load(path / "pyro_store", map_location=device)
    dictionary = torch.load(path / "dictionary")
    labels = torch.load(path / "labels")

    train_data = torch.load(path / "train_data", map_location=device)
    train_labels = torch.load(path / "train_labels", map_location=device)

    return slda, dictionary, labels, train_data, train_labels


def retrain_model(path: Path) -> SLDA:
    import gc

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    train_data: torch.Tensor = torch.load(path / "train_data", map_location=device)
    train_labels: torch.Tensor = torch.load(path / "train_labels", map_location=device)

    pyro.get_param_store().clear()

    slda = SLDA(
        voc_size=train_data.shape[-1],
        label_size=train_labels.shape[-1],
        num_topics=50,
        observe_negative_labels=torch.tensor(True, device=device),
    ).to(device)

    slda.run_svi(
        train_args=[train_data, train_labels],
        train_data_len=train_data.shape[0],
        elbo=pyro.infer.TraceEnum_ELBO(num_particles=3, vectorize_particles=True),
        initial_lr=0.01,
    )

    torch.save(slda, path / "slda")
    pyro.get_param_store().save(path / "pyro_store_slda")

    return slda
