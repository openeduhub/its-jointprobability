from __future__ import annotations

import math
from abc import abstractmethod
from collections import deque
from collections.abc import Callable, Collection, Iterator
from typing import Any, Optional, Sequence

import numpy as np
import pyro
import pyro.infer
import pyro.optim
import torch
from its_jointprobability.utils import (
    batch_to_list,
    get_random_batch_strategy,
    get_sequential_batch_strategy,
    device,
)
from pyro.nn.module import PyroModule
from tqdm import tqdm, trange


def default_data_loader(
    *tensors: torch.Tensor, batch_size: Optional[int] = None
) -> Iterator[tuple[bool, list[torch.Tensor]]]:
    n = len(tensors[0])
    if batch_size is None:
        batch_size = min(3000, max(math.ceil(n ** (3 / 4)), min(1000, n)))

    batch_strategy = get_random_batch_strategy(n, batch_size)

    for last_batch_in_epoch, batch in batch_strategy:
        yield last_batch_in_epoch, [
            tensor[batch].to(device).float() for tensor in tensors
        ]
        
def sequential_data_loader(
    *tensors: torch.Tensor, batch_size: Optional[int] = None
) -> Iterator[tuple[bool, list[torch.Tensor]]]:
    n = len(tensors[0])
    if batch_size is None:
        batch_size = min(3000, max(math.ceil(n ** (3 / 4)), min(1000, n)))

    batch_strategy = get_sequential_batch_strategy(n, batch_size)

    for last_batch_in_epoch, batch in batch_strategy:
        yield last_batch_in_epoch, [
            tensor[batch].to(device).float() for tensor in tensors
        ]


class Simple_Model:
    """Base class for Bayesian models that are learned through SVI."""

    # the random variables that are contained within posterior samples
    return_sites: Collection[str]
    # the dimensions along which to concat the random variables during batching
    return_site_cat_dim: dict[str, int]

    svi_pre_hooks: Collection[Callable[[], Any]]
    svi_step_hooks: Collection[Callable[[], Any]]
    svi_post_hooks: Collection[Callable[[], Any]]
    svi_self_pre_hooks: Collection[Callable[[Simple_Model], Any]]
    svi_self_step_hooks: Collection[Callable[[Simple_Model], Any]]
    svi_self_post_hooks: Collection[Callable[[Simple_Model], Any]]

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
        data_loader: Iterator[tuple[bool, Sequence]],
        initial_lr: float = 0.1,
        gamma: float = 0.1,
        betas: tuple[float, float] = (0.95, 0.999),
        min_epochs: int = 100,
        max_epochs: int = 1000,
        min_rel_std: float = 0.01,
        z_score_num: int = 10,
        min_z_score: float = 1.0,
        keep_prev_losses: bool = False,
    ) -> None:
        """
        Run stochastic variational inference.

        :param gamma: The decay rate of the learning rate.
            After 100 epochs, the learning rate will be multiplied with gamma.
            Note that this effect is multiplicative, so after 200 epochs,
            the learning rate will be multiplied with gamma^2,
            after 300 epochs, it will be multiplied with gamma^3, and so on.
        """

        for hook in self.svi_pre_hooks:
            hook()
        for hook in self.svi_self_pre_hooks:
            hook(self)

        if not hasattr(self, "losses") or not keep_prev_losses:
            self.losses = list()

        optim = pyro.optim.ClippedAdam(
            {
                # initial learning rate
                "lr": initial_lr,
                # the decay of the learning rate per training step
                "lrd": gamma ** (1 / 100),
                # hyperparameters for the per-parameter momentum
                "betas": betas,
            }
        )
        svi = pyro.infer.SVI(self.model, self.guide, optim, elbo)

        # collect the last z-scores using a ring buffer
        z_scores = deque(maxlen=z_score_num)

        # progress bar / iterator over epochs
        epochs = trange(max_epochs, desc="svi steps", miniters=10)
        for epoch in epochs:
            batch_losses = list()
            for last_batch_in_epoch, train_args in data_loader:
                batch_losses.append(svi.step(*train_args))
                # break if this was the last batch
                if last_batch_in_epoch:
                    break

            loss = np.mean(batch_losses)
            self.losses.append(loss)

            for hook in self.svi_step_hooks:
                hook()
            for hook in self.svi_self_step_hooks:
                hook(self)

            # compute the last z-score
            mean = np.mean(self.losses[-z_score_num:])
            std = np.std(self.losses[-z_score_num:])
            rel_std = np.abs(std / mean)  # type: ignore
            z_scores.append(np.abs((self.losses[-1] - mean) / std))

            epochs.set_postfix(
                epoch_loss=f"{self.losses[-1]:.2e}", z_score=f"{z_scores[-1]:.2f}"
            )

            if (
                epoch > min_epochs
                and all(z_score < min_z_score for z_score in z_scores)
                and rel_std < min_rel_std
            ):
                break

        for hook in self.svi_post_hooks:
            hook()
        for hook in self.svi_self_post_hooks:
            hook(self)

    def predictive(self, *args, **kwargs) -> pyro.infer.Predictive:
        """Return a Predictive object in order to generate posterior samples."""
        return pyro.infer.Predictive(self.model, guide=self.guide, *args, **kwargs)

    def draw_posterior_samples(
        self,
        data_len: int,
        data_args: Optional[Collection[Any]] = None,
        num_samples: int = 100,
        parallel_sample: bool = False,
        batch_size: int = 1000,
        return_sites: Optional[Collection[str]] = None,
        progress_bar: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Draw posterior samples from this model."""
        return_sites = return_sites if return_sites is not None else self.return_sites
        data_args = data_args if data_args is not None else []
        batch_size = min(data_len, batch_size)
        predictive = self.predictive(
            num_samples=num_samples, return_sites=return_sites, parallel=parallel_sample
        )

        posterior_samples = None
        # draw from the posterior in batches
        batch_strategy = sequential_data_loader(*data_args, batch_size=batch_size)
        batches = batch_to_list(batch_strategy)
        if progress_bar:
            bar = tqdm(batches, desc="posterior sample")
        else:
            bar = batches

        for batch in bar:
            with torch.no_grad():
                posterior_batch: dict[str, torch.Tensor] = predictive(*batch)

            if posterior_samples is None:
                posterior_samples = posterior_batch

            else:
                for key in posterior_samples.keys():
                    posterior_samples[key] = torch.cat(
                        [posterior_samples[key], posterior_batch[key]],
                        dim=self.return_site_cat_dim[key],
                    )
                    del posterior_batch[key]

        if posterior_samples is None:
            raise ValueError("Cannot sample for empty data!")

        return self.clean_up_posterior_samples(posterior_samples)


class Model(Simple_Model, PyroModule):
    """A Bayesian model that relies on a neural network."""

    def __init__(self) -> None:
        self.svi_pre_hooks = [self.train]
        self.svi_self_pre_hooks = []
        self.svi_step_hooks = []
        self.svi_self_step_hooks = []
        self.svi_post_hooks = [self.eval]
        self.svi_self_post_hooks = []
        super().__init__()
