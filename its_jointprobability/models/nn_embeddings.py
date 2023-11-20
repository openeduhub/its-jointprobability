from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Optional

import pyro
import pyro.distributions as dist
from pyro.infer import Predictive
import pyro.infer.autoguide
import pyro.optim
from pyro.infer.energy_distance import reduce
import torch
from tqdm import trange
from tyxe.likelihoods import PyroModule, PyroSample
from its_jointprobability.models.model import Model
from its_jointprobability.utils import Quality_Result, device, quality_measures
from its_jointprobability.models.prodslda_tyxe import get_bayes_encoder
import torch.nn as nn


class Embeddings_NN(PyroModule):
    def __init__(
        self,
        emb_dim: int,
        out_dim: int,
        dropout: Optional[float] = None,
        hidden_layers: tuple[int, ...] = (100, 100),
        observe_negative_labels=torch.tensor(True),
        weight_locs: Optional[Sequence[torch.Tensor]] = None,
        weight_stds: Optional[Sequence[torch.Tensor]] = None,
        bias_locs: Optional[Sequence[torch.Tensor]] = None,
        bias_stds: Optional[Sequence[torch.Tensor]] = None,
    ):
        super().__init__()
        self.observe_negative_labels = observe_negative_labels
        self.activation = nn.Softplus()
        self.emb_dim = emb_dim
        self.out_dim = out_dim

        layers = (emb_dim,) + hidden_layers + (2 * out_dim,)

        if weight_locs is None:
            weight_locs = [
                torch.zeros([layer_out, layer_in])
                for layer_in, layer_out in zip(layers, layers[1:])
            ]
        if weight_stds is None:
            weight_stds = [
                torch.ones([layer_out, layer_in]) * 10
                for layer_in, layer_out in zip(layers, layers[1:])
            ]
        if bias_locs is None:
            bias_locs = [
                torch.zeros([layer_out])
                for layer_in, layer_out in zip(layers, layers[1:])
            ]
        if bias_stds is None:
            bias_stds = [
                torch.ones([layer_out]) * 10
                for layer_in, layer_out in zip(layers, layers[1:])
            ]

        self.layers = list()
        for index, (
            layer_in,
            layer_out,
            weight_loc,
            weight_std,
            bias_loc,
            bias_std,
        ) in enumerate(
            zip(layers, layers[1:], weight_locs, weight_stds, bias_locs, bias_stds)
        ):
            # avoid duplicate site names
            setattr(self, f"module_{index}", PyroModule[nn.Linear](layer_out, layer_in))
            getattr(self, f"module_{index}").weight = PyroSample(
                dist.Normal(weight_loc, weight_std).to_event(2)
            )
            getattr(self, f"module_{index}").bias = PyroSample(
                dist.Normal(bias_loc, bias_std).to_event(1)
            )
            self.layers.append(getattr(self, f"module_{index}"))

        if dropout is not None:
            self.layers.insert(-2, nn.Dropout(dropout))

    def forward(self, inputs: torch.Tensor, obs: Optional[torch.Tensor] = None):
        h = reduce(lambda x, layer: self.activation(layer(x)), self.layers, inputs)
        loga_loc, loga_logvar = h.split(self.out_dim, -1)
        loga_scale = nn.functional.softmax(loga_logvar, -1)
        with pyro.plate("documents", inputs.shape[-2], dim=-2):
            with pyro.plate("labels", self.out_dim, dim=-1):
                loga = pyro.sample("loga", dist.Normal(loga_loc, loga_scale))
                pyro.sample(
                    "obs",
                    dist.Bernoulli(logits=loga),
                    obs=obs,
                    obs_mask=torch.logical_or(
                        self.observe_negative_labels,
                        obs.bool() if obs is not None else torch.tensor(False),
                    ),
                    infer={"enumerate": "parallel"},
                )


def retrain_model(path: Path) -> Predictive:
    torch.set_default_device(device)
    train_data: torch.Tensor = torch.load(
        path / "train_data_embeddings", map_location=device
    ).float()
    train_labels: torch.Tensor = torch.load(
        path / "train_labels_embeddings", map_location=device
    ).float()

    pyro.clear_param_store()
    pyro.set_rng_seed(0)

    model = Embeddings_NN(
        out_dim=train_labels.shape[-1],
        emb_dim=train_data.shape[-1],
        hidden_layers=(128, 256, 128),
        observe_negative_labels=torch.tensor(True),
    )

    # guide = pyro.infer.autoguide.AutoDiagonalNormal(
    #     pyro.poutine.block(model, hide=["obs_unobserved"])
    # )

    # elbo = pyro.infer.TraceEnum_ELBO(num_particles=3)
    # optim = pyro.optim.ClippedAdam({"lr": 0.1, "lrd": 0.999})

    # svi = pyro.infer.SVI(model, guide, optim, elbo)
    # bar = trange(1000)
    # for _ in bar:
    #     loss = svi.step(train_data, train_labels)
    #     bar.set_postfix(loss=f"{loss:.3e}")
    # bar.close()

    kernel = pyro.infer.NUTS(model, jit_compile=False)
    mcmc = pyro.infer.MCMC(kernel, num_samples=100, warmup_steps=100)
    mcmc.run(train_data, train_labels)

    return Predictive(model=model, posterior_samples=mcmc.get_samples())


def train_here_qualities() -> Quality_Result:
    path = Path.cwd() / "data"
    pred = retrain_model(path)

    train_data: torch.Tensor = torch.load(
        path / "train_data_embeddings", map_location=device
    ).float()
    train_labels: torch.Tensor = torch.load(
        path / "train_labels_embeddings", map_location=device
    ).float()

    samples = pred(train_data)["obs"]
    return quality_measures(samples, train_labels, mean_dim=0, cutoff=None)
