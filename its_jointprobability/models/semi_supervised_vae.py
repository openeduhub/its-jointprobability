from __future__ import annotations

import argparse
import math
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Iterator, Optional

import matplotlib.pyplot as plt
import pandas as pd
import pyro.distributions as dist
import pyro.infer
import pyro.optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic
from its_jointprobability.models.model import Model
from its_jointprobability.utils import (
    Quality_Result,
    device,
    get_random_batch_strategy,
    quality_measures,
    sequential_data_loader,
)
from pyro.infer.enum import partial
from pyro.nn.module import to_pyro_module_


class Semi_Supervised_VAE(Model):
    return_sites = ("y", "z")
    return_site_cat_dim = {"y": -1, "z": -1}

    def __init__(
        self,
        vocab_dim: int,
        label_dim: int,
        z_dim: int = 100,
        hid_dim: int = 100,
        dropout: float = 0.2,
        svi_step_self_hooks: Optional[
            Iterable[Callable[[Semi_Supervised_VAE], Any]]
        ] = None,
    ):
        super().__init__()
        self.vocab_dim = vocab_dim
        self.label_dim = label_dim
        self.z_dim = z_dim

        self.decoder = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(label_dim + z_dim, vocab_dim),
            # nn.BatchNorm1d(vocab_dim, affine=False),
            nn.Sigmoid(),
        )
        to_pyro_module_(self.decoder)

        self.encoder_y = nn.Sequential(
            nn.Linear(vocab_dim, hid_dim),
            nn.Softplus(),
            nn.Linear(hid_dim, hid_dim),
            nn.Softplus(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, label_dim),
            # nn.BatchNorm1d(label_dim * 2, affine=False),
            nn.Sigmoid(),
        )
        to_pyro_module_(self.encoder_y)

        self.encoder_z = nn.Sequential(
            nn.Linear(vocab_dim + label_dim, hid_dim),
            nn.Softplus(),
            nn.Linear(hid_dim, hid_dim),
            nn.Softplus(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, z_dim * 2),
            # nn.BatchNorm1d(z_dim, affine=False),
        )
        to_pyro_module_(self.encoder_z)

        self.accuracies = list()
        self.losses = list()

        if svi_step_self_hooks is not None:
            self.svi_step_hooks = list(self.svi_step_hooks) + [
                partial(hook, self) for hook in svi_step_self_hooks
            ]

    def model(self, xs: torch.Tensor, ys: Optional[torch.Tensor] = None):
        ic()
        n = xs.shape[-2]

        with pyro.plate("data", n, dim=-1):
            prior_loc = xs.new_zeros([self.z_dim])
            prior_scale = xs.new_ones([self.z_dim])
            zs = pyro.sample("z", dist.Normal(prior_loc, prior_scale).to_event(1))
            ic(zs.shape)

            alpha_prior = xs.new_ones([self.label_dim]) / self.label_dim
            ic(alpha_prior.shape)

            ys = pyro.sample(
                "y",
                dist.Bernoulli(alpha_prior).to_event(1),  # type: ignore
                obs=ys,
            )
            ic(ys.shape)

            assert ys is not None

            word_probs = self.decoder(torch.concat([zs, ys], dim=-1))
            ic(word_probs.shape)

            # use the maximum document length
            # this is fine, because the count number does not affect
            # the log prob of the multinomial
            total_count = int(xs.sum(-1).max())
            xs = pyro.sample("x", dist.Multinomial(total_count, word_probs), obs=xs)
            ic(xs.shape)

    @pyro.infer.config_enumerate
    def guide(self, xs: torch.Tensor, ys: Optional[torch.Tensor] = None):
        ic()
        n = xs.shape[-2]
        with pyro.plate("data", n, dim=-1):
            if ys is None:
                alpha = self.encoder_y(xs)
                ys = pyro.sample(
                    "y",
                    dist.Bernoulli(alpha).to_event(1),  # type: ignore
                )

            assert ys is not None
            ic(ys.shape)

            loc, logscale = self.encoder_z(torch.concat([xs, ys], dim=-1)).split(
                self.z_dim, -1
            )
            scale = F.softplus(logscale)
            ic(loc.shape)
            ic(scale.shape)
            zs = pyro.sample("z", dist.Normal(loc, scale).to_event(1))
            ic(zs.shape)

    def calculate_accuracy(
        self, docs: torch.Tensor, labels: torch.Tensor, batch_size: Optional[int] = None
    ) -> float:
        return quality_measures(
            self.draw_posterior_samples(
                data_loader=sequential_data_loader(
                    docs,
                    batch_size=batch_size,
                    device=device,
                    dtype=torch.float,
                ),
                num_samples=1,
                return_sites=["label"],
                progress_bar=False,
            )["label"],
            labels=labels,
            cutoff=1.0,
        ).accuracy  # type: ignore

    def append_to_accuracies_(
        self, docs: torch.Tensor, labels: torch.Tensor, batch_size: Optional[int] = None
    ) -> None:
        self.accuracies.append(self.calculate_accuracy(docs, labels, batch_size))


def retrain_model(
    path: Path, n=None, seed=0, svi_step_self_hooks=None, **kwargs
) -> Semi_Supervised_VAE:
    data = import_data(path)
    train_data_labeled: torch.Tensor = data["train_data_labeled"]
    train_data_unlabeled: torch.Tensor = data["train_data_unlabeled"]
    train_labels: torch.Tensor = data["train_labels"]
    pyro.set_rng_seed(seed)

    if n is not None:
        train_data_labeled = train_data_labeled[:n]
        train_labels = train_labels[:n]

    pyro.get_param_store().clear()

    model = Semi_Supervised_VAE(
        vocab_dim=train_data_labeled.shape[-1],
        label_dim=train_labels.shape[-1],
        hid_dim=50,
        z_dim=100,
        dropout=0.2,
        svi_step_self_hooks=svi_step_self_hooks,
    ).to(device)

    def data_loader() -> Iterator[tuple[bool, list[torch.Tensor]]]:
        n_labeled = len(train_data_labeled)
        n_unlabeled = len(train_data_unlabeled)
        batch_size_labeled = min(
            3000, max(math.ceil(n_labeled ** (3 / 4)), min(1000, n_labeled))
        )
        batch_size_unlabeled = min(
            3000, max(math.ceil(n_unlabeled ** (3 / 4)), min(1000, n_unlabeled))
        )
        batch_strategy_labeled = get_random_batch_strategy(
            n_labeled, batch_size_labeled
        )
        batch_strategy_unlabeled = get_random_batch_strategy(
            n_unlabeled, batch_size_unlabeled
        )

        visited_labeled = 0
        visited_unlabeled = 0

        while True:
            if visited_labeled / n_labeled >= visited_unlabeled / n_unlabeled:
                _, batch_ids = batch_strategy_unlabeled.__next__()
                visited_unlabeled += len(batch_ids)
                # move batch to GPU, if using, and convert to floats
                batch = [train_data_unlabeled[batch_ids].to(device).float()]
            else:
                _, batch_ids = batch_strategy_labeled.__next__()
                visited_labeled += len(batch_ids)
                # move batch to GPU, if using, and convert to floats
                batch = [
                    tensor[batch_ids].to(device).float()
                    for tensor in [train_data_labeled, train_labels]
                ]

            if visited_labeled == n_labeled and visited_unlabeled == n_unlabeled:
                visited_labeled = 0
                visited_unlabeled = 0
                last_batch_of_epoch = True
            else:
                last_batch_of_epoch = False

            ic(batch[0].shape)
            yield last_batch_of_epoch, batch

    model.run_svi(
        elbo=pyro.infer.TraceEnum_ELBO(num_particles=1, max_plate_nesting=1),
        data_loader=data_loader(),
        max_epochs=500,
        **kwargs,
    )

    model = model.eval()
    torch.save(model, path / "semi-supervised_vae")

    return model


def import_data(path: Path) -> dict[str, Any]:
    train_data_labeled: torch.Tensor = torch.load(
        path / "train_data_labeled", map_location=torch.device("cpu")
    )
    train_labels: torch.Tensor = torch.load(
        path / "train_labels", map_location=torch.device("cpu")
    )
    train_data_unlabeled: torch.Tensor = torch.load(
        path / "train_data_unlabeled", map_location=torch.device("cpu")
    )

    return {
        "train_data_labeled": train_data_labeled,
        "train_data_unlabeled": train_data_unlabeled,
        "train_labels": train_labels,
    }


def retrain_model_cli():
    """Add some CLI arguments to the retraining of the model."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        type=str,
        help="The path to the directory containing the training data",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="The seed to use for pseudo random number generation",
    )
    parser.add_argument("--plot", action="store_true")

    ic.disable()

    args = parser.parse_args()

    path = Path(args.path)
    data = import_data(path)
    train_data_labeled: torch.Tensor = data["train_data_labeled"]
    train_labels: torch.Tensor = data["train_labels"]

    model = retrain_model(
        Path(args.path),
        seed=args.seed,
        # if plotting the training process,
        # calculate the training data accuracy after every epoch
        svi_step_self_hooks=[
            (
                lambda x: x.append_to_accuracies_(
                    docs=train_data_labeled,
                    labels=train_labels,
                    batch_size=None,
                )
            )
        ]
        if args.plot
        else None,
        keep_prev_losses=True,
    )

    if args.plot:
        fig = plt.figure(figsize=(16, 9), dpi=100)
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.plot(model.losses)
        ax1.set_yscale("symlog")
        ax1.set_title("Loss function")
        ax1.set_ylabel("Negative ELBO")
        ax2 = fig.add_subplot(2, 1, 2)
        ax2.plot(model.accuracies)
        ax2.set_yscale("symlog")
        ax2.set_title("Training set accuracy")
        ax2.set_xlabel("Training epoch")
        ax2.set_ylabel("Accuracy")

        fig.savefig("./training_process.png")

    labels: torch.Tensor = torch.load(path / "labels")

    # evaluate the newly trained model on the training data
    print("evaluating model on train data")
    eval_model(model, train_data_labeled, train_labels, labels)

    try:
        # evaluate the newly trained model on the testing data
        print("evaluating model on test data")
        test_data_labeled: torch.Tensor = torch.load(
            path / "test_data_labeled", map_location=torch.device("cpu")
        )
        test_labels: torch.Tensor = torch.load(
            path / "test_labels", map_location=torch.device("cpu")
        )

        eval_model(model, test_data_labeled, test_labels, labels)
    except FileNotFoundError:
        pass


def eval_model(
    model: Semi_Supervised_VAE,
    data: torch.Tensor,
    labels: torch.Tensor,
    label_values: Iterable,
) -> Quality_Result:
    ic.disable()
    samples = model.draw_posterior_samples(
        data_loader=sequential_data_loader(
            data,
            device=device,
            dtype=torch.float,
        ),
        return_sites=["a"],
        num_samples=100,
    )["a"]
    global_measures = quality_measures(samples, labels, mean_dim=0, cutoff=None)
    print(f"global measures: {global_measures}")

    by_discipline = quality_measures(
        samples,
        labels,
        mean_dim=-3,
        cutoff=global_measures.cutoff,
        parallel_dim=-1,
    )
    df = pd.DataFrame(
        {
            key: getattr(by_discipline, key)
            for key in ["accuracy", "precision", "recall", "f1_score"]
        }
    )
    df["taxonid"] = label_values
    df["count"] = labels.sum(-2).cpu()
    df = df.set_index("taxonid")
    print(df.sort_values("f1_score", ascending=False))

    return by_discipline
