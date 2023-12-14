import math
from abc import abstractmethod
from collections import deque
from collections.abc import Callable, Collection, Iterable, Iterator
from typing import Any, Optional

import numpy as np
import pandas as pd
import pyro
import pyro.infer
import pyro.optim
import torch
import torch.nn as nn
from icecream import ic
from its_jointprobability.utils import (
    Quality_Result,
    batch_to_list,
    get_random_batch_strategy,
    get_sequential_batch_strategy,
    quality_measures,
)
from tqdm import tqdm, trange


class Simple_Model:
    """Base class for Bayesian models that are learned through SVI."""

    # the random variables that are contained within posterior samples
    return_sites: Collection[str]
    # the dimensions along which to concat the random variables during batching
    return_site_cat_dim: dict[str, int]

    @abstractmethod
    def model(self, *args, batch: Optional[Collection[int]] = None, **kwargs):
        """The prior model"""
        ...

    @abstractmethod
    def guide(self, *args, batch: Optional[Collection[int]] = None, **kwargs):
        """The variational family that approximates the posterior on latent RVs"""
        ...

    def clean_up_posterior_samples(
        self, posterior_samples: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Override this method in order to clean up the posterior samples"""
        return posterior_samples

    def run_svi(
        self,
        elbo: pyro.infer.ELBO,
        train_data_len: int,
        train_args: Optional[Collection[Any]] = None,
        train_kwargs: Optional[dict[str, Any]] = None,
        batch_strategy_factory: Optional[
            Callable[[int, int], Iterator[tuple[bool, Collection[int]]]]
        ] = None,
        batch_size: Optional[int] = None,
        initial_lr: float = 0.1,
        gamma: float = 0.1,
        betas: tuple[float, float] = (0.95, 0.999),
        min_epochs: int = 100,
        max_epochs: int = 1000,
        min_rel_std: float = 0.01,
        z_score_num: int = 10,
        min_z_score: float = 1.0,
    ) -> list[float]:
        """
        Run stochastic variational inference.

        Returns
        -------
        The list of losses
        """
        train_args = train_args or list()
        train_kwargs = train_kwargs or dict()

        if batch_size is None:
            batch_size = min(
                3000,
                max(
                    math.ceil(train_data_len ** (3 / 4)),
                    min(1000, train_data_len),
                ),
            )

        ic(batch_size)

        batches_per_epoch = math.ceil(train_data_len / batch_size)
        optim = pyro.optim.ClippedAdam(
            {
                # initial learning rate
                "lr": initial_lr,
                # final learning rate will be gamma * initial_lr
                "lrd": gamma ** (1 / (max_epochs * batches_per_epoch)),
                # hyperparameters for the per-parameter momentum
                "betas": betas,
            }
        )
        svi = pyro.infer.SVI(self.model, self.guide, optim, elbo)
        if batch_strategy_factory is None:
            batch_strategy = get_random_batch_strategy(train_data_len, batch_size)
        else:
            batch_strategy = batch_strategy_factory(train_data_len, batch_size)

        losses = list()
        # collect the last z-scores using a ring buffer
        z_scores = deque(maxlen=z_score_num)

        # progress bar / iterator over epochs
        epochs = trange(max_epochs, desc="svi steps", miniters=10)
        for epoch in epochs:
            batch_losses = list()
            for last_batch_in_epoch, minibatch in batch_strategy:
                # if a batch size has been given,
                # assume that the model and guide
                # are parameterized by a subsample index list
                batch_losses.append(
                    svi.step(*train_args, batch=minibatch, **train_kwargs)
                )
                # break if this was the last batch
                if last_batch_in_epoch:
                    break

            loss = np.mean(batch_losses)
            losses.append(loss)

            # compute the last z-score
            mean = np.mean(losses[-z_score_num:])
            std = np.std(losses[-z_score_num:])
            rel_std = np.abs(std / mean)  # type: ignore
            z_scores.append(np.abs((losses[-1] - mean) / std))

            epochs.set_postfix(
                epoch_loss=f"{losses[-1]:.2e}", z_score=f"{z_scores[-1]:.2f}"
            )

            if (
                epoch > min_epochs
                and all(z_score < min_z_score for z_score in z_scores)
                and rel_std < min_rel_std
            ):
                break

        return losses

    def predictive(self, *args, **kwargs) -> pyro.infer.Predictive:
        """Return a Predictive object in order to generate posterior samples."""
        return pyro.infer.Predictive(self.model, guide=self.guide, *args, **kwargs)

    def draw_posterior_samples(
        self,
        data_len: int,
        data_args: Optional[list[Any]] = None,
        data_kwargs: Optional[dict[str, Any]] = None,
        num_samples: int = 100,
        parallel_sample: bool = False,
        batch_size: int = 1000,
        return_sites: Optional[Collection[str]] = None,
    ) -> dict[str, torch.Tensor]:
        """Draw posterior samples from this model."""
        return_sites = return_sites if return_sites is not None else self.return_sites
        data_args = data_args if data_args is not None else list()
        data_kwargs = data_kwargs if data_kwargs is not None else dict()
        batch_size = min(data_len, batch_size)

        batch_strategy = get_sequential_batch_strategy(data_len, batch_size)
        predictive = self.predictive(
            num_samples=num_samples, return_sites=return_sites, parallel=parallel_sample
        )

        posterior_samples = None
        # draw from the posterior in batches
        for batch in tqdm(batch_to_list(batch_strategy), desc="posterior sample"):
            with torch.no_grad():
                posterior_batch: dict[str, torch.Tensor] = predictive(
                    *data_args, batch=batch, **data_kwargs
                )

            if posterior_samples is None:
                posterior_samples = posterior_batch

            else:
                for key in posterior_samples.keys():
                    ic(posterior_batch[key].shape)
                    posterior_samples[key] = torch.cat(
                        [posterior_samples[key], posterior_batch[key]],
                        dim=self.return_site_cat_dim[key],
                    )
                    del posterior_batch[key]

        if posterior_samples is None:
            raise ValueError("Cannot sample for empty data!")

        return self.clean_up_posterior_samples(posterior_samples)


class Model(Simple_Model, nn.Module):
    """A Bayesian model that relies on a neural network."""

    def run_svi(self, *args, **kwargs) -> list[float]:
        self.train()
        return super().run_svi(*args, **kwargs)

    def draw_posterior_samples(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        self.eval()
        return super().draw_posterior_samples(*args, **kwargs)


def eval_model(
    model: Simple_Model,
    data: torch.Tensor,
    labels: torch.Tensor,
    label_values: Iterable,
    site="a",
) -> Quality_Result:
    ic.disable()
    samples = model.draw_posterior_samples(
        data.shape[-2], data_args=[data], return_sites=[site], num_samples=100
    )[site]
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
