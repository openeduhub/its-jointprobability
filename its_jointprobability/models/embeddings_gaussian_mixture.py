from collections.abc import Collection
from pathlib import Path
from typing import Optional

import pyro
import pyro.distributions as dist
import pyro.infer.autoguide
import torch
from its_jointprobability.models.model import Model
from its_jointprobability.utils import device


class Embeddings_Gaussian_Mixture(Model):
    return_sites = ("label", "nu", "a")
    return_site_cat_dim = {"nu": 0, "a": -1, "label": -1}

    def __init__(
        self,
        label_size: int,
        embedding_dims: int,
        nu_loc: float = 0.0,
        nu_scale: float = 10.0,
        observe_negative_labels=torch.tensor(True),
    ):
        super().__init__()
        self.label_size = label_size
        self.embedding_dims = embedding_dims
        self.nu_loc = nu_loc
        self.nu_scale = nu_scale
        self.observe_negative_labels = observe_negative_labels

    @pyro.infer.config_enumerate()
    def model(
        self,
        docs: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        batch: Optional[Collection[int]] = None,
    ):
        labels_plate = pyro.plate("labels", self.label_size, dim=-2)

        # the label application coefficients
        with labels_plate:
            nu = pyro.sample(
                "nu",
                dist.Normal(
                    self.nu_loc * docs.new_ones(self.embedding_dims),
                    self.nu_scale * docs.new_ones(self.embedding_dims),
                ).to_event(1),
            )
        with pyro.plate("documents", docs.shape[-2], subsample=batch, dim=-1) as ind:
            with labels_plate:
                a = pyro.sample("a", dist.Normal(nu.squeeze(-2) @ docs[ind].T, 10))
                label = pyro.sample(
                    "label",
                    dist.Bernoulli(logits=a),  # type: ignore
                    obs=labels[ind].T if labels is not None else None,
                    obs_mask=torch.logical_or(
                        self.observe_negative_labels.to(device), labels[ind].T.bool()
                    )
                    if labels is not None
                    else None,
                )

    def guide(
        self,
        docs: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        batch: Optional[Collection[int]] = None,
    ):
        labels_plate = pyro.plate("labels", self.label_size, dim=-2)

        # the label application coefficients
        with labels_plate:
            nu_q_loc = pyro.param(
                "nu_q_loc",
                lambda: torch.randn(
                    [self.label_size, self.embedding_dims],
                    device=docs.device,
                ).unsqueeze(-2),
            )
            nu_q_scale = pyro.param(
                "nu_q_scale",
                lambda: docs.new_ones(
                    [self.label_size, self.embedding_dims],
                ).unsqueeze(-2),
                constraint=dist.constraints.positive,
            )
            nu_q = pyro.sample(
                "nu",
                dist.Normal(
                    nu_q_loc,
                    nu_q_scale,
                ).to_event(1),
            )

        with pyro.plate("documents", docs.shape[-2], subsample=batch, dim=-1) as ind:
            with labels_plate:
                a_q_scale = pyro.param(
                    "a_q_scale",
                    lambda: docs.new_ones([self.label_size]).unsqueeze(-1),
                    constraint=dist.constraints.positive,
                )
                a_q = pyro.sample(
                    "a", dist.Normal(nu_q.squeeze(-2) @ docs[ind].T, a_q_scale)
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


def retrain_model(path: Path) -> Embeddings_Linear_Regression:
    train_data: torch.Tensor = torch.load(
        path / "train_data_embeddings", map_location=device
    ).float()
    train_labels: torch.Tensor = torch.load(
        path / "train_labels_embeddings", map_location=device
    ).float()

    pyro.get_param_store().clear()

    embedding_model = Embeddings_Linear_Regression(
        label_size=train_labels.shape[-1], embedding_dims=train_data.shape[-1]
    ).to(device)

    embedding_model.run_svi(
        train_args=[train_data, train_labels],
        train_data_len=train_data.shape[0],
        elbo=pyro.infer.TraceEnum_ELBO(num_particles=5, vectorize_particles=True),
    )

    torch.save(embedding_model, path / "embedding")
    pyro.get_param_store().save(path / "pyro_store_embedding")

    return embedding_model
