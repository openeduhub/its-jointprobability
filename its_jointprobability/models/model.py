from __future__ import annotations

from abc import abstractmethod
from collections import deque
from collections.abc import Callable, Collection, Iterable, Sequence
from typing import Any, Optional

import numpy as np
import pyro
import pyro.infer
import pyro.optim
import torch
from its_jointprobability.utils import Data_Loader, batch_to_list
from pyro.nn.module import PyroModule
from tqdm import tqdm, trange
from icecream import ic


class Simple_Model:
    """Base class for Bayesian models that are learned through SVI."""

    # the random variables that are contained within posterior samples
    return_sites: Collection[str]
    # the dimensions along which to concat the random variables during batching
    return_site_cat_dim: dict[str, int]

    # various hooks that may be run during SVI
    svi_pre_hooks: Collection[Callable[[], Any]]  # before start
    svi_step_hooks: Collection[Callable[[], Any]]  # after each step
    svi_post_hooks: Collection[Callable[[], Any]]  # after svi is done
    # like the hooks above, but these are passed the Model object
    svi_self_pre_hooks: Collection[Callable[[Simple_Model], Any]]
    svi_self_step_hooks: Collection[Callable[[Simple_Model], Any]]
    svi_self_post_hooks: Collection[Callable[[Simple_Model], Any]]

    @abstractmethod
    def model(self, *args, **kwargs):
        """The prior model"""
        ...

    @abstractmethod
    def guide(self, *args, **kwargs):
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
        data_loader: Data_Loader,
        initial_lr: float = 0.1,
        gamma: float = 0.5,
        betas: tuple[float, float] = (0.9, 0.999),
        min_epochs: int = 100,
        max_epochs: int = 1000,
        min_rel_std: float = 0.01,
        min_z_score: float = 1.0,
        metric_len: int = 10,
        keep_prev_losses: bool = False,
    ) -> None:
        """
        Run stochastic variational inference on the data given by the data loader.

        :param gamma: The decay rate of the learning rate.
            After 100 epochs, the learning rate will be multiplied with gamma.
            Note that this effect is multiplicative, so after 200 epochs,
            the learning rate will be multiplied with gamma^2,
            after 300 epochs, it will be multiplied with gamma^3, and so on.
        """

        for hook in self.svi_pre_hooks:
            hook()
        for self_hook in self.svi_self_pre_hooks:
            self_hook(self)

        if not hasattr(self, "losses") or not keep_prev_losses:
            self.losses = list()

        one_epoch = batch_to_list(data_loader)
        batches_per_epoch = len(one_epoch)
        n = sum(len(batch[0]) for batch in one_epoch)

        lrd = gamma ** (1 / (100 * batches_per_epoch))
        ic(lrd)
        optim = pyro.optim.ClippedAdam(
            {
                # initial learning rate
                "lr": initial_lr,
                # the decay of the learning rate per training step
                "lrd": lrd,
                # hyperparameters for the per-parameter momentum
                "betas": betas,
            }
        )
        svi = pyro.infer.SVI(self.model, self.guide, optim, elbo)

        # progress bar / iterator over epochs
        epochs = trange(max_epochs, desc="svi steps", miniters=10)
        for epoch in epochs:
            batch_losses = list()
            for last_batch_in_epoch, train_args in data_loader:
                with pyro.poutine.scale(scale=n / len(train_args[0])):
                    batch_losses.append(svi.step(*train_args))
                # break if this was the last batch
                if last_batch_in_epoch:
                    break

            loss = np.mean(batch_losses)
            self.losses.append(loss)

            for hook in self.svi_step_hooks:
                hook()
            for self_hook in self.svi_self_step_hooks:
                self_hook(self)

            # compute the metrics to determine whether to stop early
            last_losses = torch.tensor(self.losses[-metric_len:])
            mean = last_losses.mean()
            std = last_losses.std()
            rel_std = torch.abs(std / mean)
            z_scores = torch.abs((last_losses - mean) / std)

            epochs.set_postfix(
                epoch_loss=f"{self.losses[-1]:.2e}",
                z_score=f"{z_scores[-1]:.2f}",
                rel_std=f"{rel_std:.2e}",
            )

            if (
                epoch > min_epochs
                and rel_std < min_rel_std
                and (z_scores < min_z_score).all()
            ):
                break

        for hook in self.svi_post_hooks:
            hook()
        for self_hook in self.svi_self_post_hooks:
            self_hook(self)

    def predictive(self, *args, **kwargs) -> pyro.infer.Predictive:
        """Return a Predictive object in order to generate posterior samples."""
        return pyro.infer.Predictive(self.model, guide=self.guide, *args, **kwargs)

    def draw_posterior_samples(
        self,
        data_loader: Data_Loader,
        num_samples: int = 100,
        parallel_sample: bool = False,
        return_sites: Optional[Collection[str]] = None,
        progress_bar: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Draw posterior samples from this model."""
        return_sites = return_sites if return_sites is not None else self.return_sites
        predictive = self.predictive(
            num_samples=num_samples, return_sites=return_sites, parallel=parallel_sample
        )

        posterior_samples = None
        # draw from the posterior in batches
        batches = batch_to_list(data_loader)
        if progress_bar:
            bar: Iterable[Sequence[torch.Tensor]] = tqdm(
                batches, desc="posterior sample"
            )
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
