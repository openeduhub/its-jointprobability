from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from pprint import pprint
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
import pyro
import pyro.infer
import pyro.optim
import torch
from icecream import ic
from its_jointprobability.utils import (
    Data_Loader,
    Quality_Result,
    batch_to_list,
    get_batch_size,
    quality_measures,
    sequential_data_loader,
    texts_to_bow_tensor,
)
from pyro.nn.module import PyroModule
from tqdm import tqdm, trange


class Simple_Model:
    """Base class for Bayesian models that are learned through SVI."""

    # the random variables that are contained within posterior samples
    return_sites: Collection[str]
    # the dimensions along which to concat the random variables during batching
    return_site_cat_dim: dict[str, int]

    # various hooks that may be run during SVI
    svi_pre_hooks: list[Callable[[], Any]]  # before start
    svi_step_hooks: list[Callable[[int, float], Any]]  # after each step
    svi_post_hooks: list[Callable[[], Any]]  # after svi is done
    # like the hooks above, but these are passed the Model object
    svi_self_pre_hooks: list[Callable[[Simple_Model], Any]]
    svi_self_step_hooks: list[Callable[[Simple_Model, int, float], Any]]
    svi_self_post_hooks: list[Callable[[Simple_Model], Any]]

    # store the device on which this model shall run
    device: Optional[torch.device] = None

    #: an annealing factor that increases to 1.0 during training.
    #: it is up to the actual model implementation to use this factor.
    annealing_factor: float = 1.0

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
        betas: tuple[float, float] = (0.90, 0.999),
        min_epochs: int = 100,
        max_epochs: int = 1000,
        annealing_epochs: int = 100,
        min_rel_std: float = 1e-3,
        min_z_score: float = 1.0,
        metric_len: int = 5,
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

        # get the number of mini-batches and total number of data points,
        # so that we can calculate the learning rate decay
        one_epoch = batch_to_list(data_loader)
        batches_per_epoch = len(one_epoch)

        # set the per-step learning rate decay such that after 100 epochs,
        # we have multiplied the learning rate with gamma
        lrd = gamma ** (1 / (100 * batches_per_epoch))
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

        # get the list of annealing factors that shall be used during training
        # at the end of annealing_epochs, factor should equal 1.0
        annealing_epochs = min(annealing_epochs, max_epochs)
        annealing_factors = torch.arange(1, annealing_epochs + 1)
        annealing_factors = (
            self.annealing_factor * (1 - annealing_factors / annealing_epochs)
            + 1.0 * annealing_factors / annealing_epochs
        )

        # progress bar / iterator over epochs
        epochs = trange(max_epochs, desc="svi steps")
        for epoch in epochs:
            if epoch < annealing_epochs:
                self.annealing_factor = float(annealing_factors[epoch])
            else:
                self.annealing_factor = 1.0

            # apply the batch-wise SVI steps
            batch_losses = list()
            for last_batch_in_epoch, batch in data_loader:
                # scale by the batch size, as the ELBO is proportional to the
                # number of data points
                batch_size = get_batch_size(batch)
                with pyro.poutine.scale(scale=1.0 / batch_size):
                    batch_losses.append(svi.step(*batch))
                if last_batch_in_epoch:
                    break

            loss = float(np.mean(batch_losses))
            self.losses.append(loss)

            for step_hook in self.svi_step_hooks:
                step_hook(epoch, loss)
            for self_step_hook in self.svi_self_step_hooks:
                self_step_hook(self, epoch, loss)

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
            bar = tqdm(batches, desc="posterior sample")
        else:
            bar = batches

        for batch in bar:
            with torch.no_grad():
                posterior_batch: dict[str, torch.Tensor] = predictive(*batch)
                # ensure that the posterior batch is on the CPU,
                # to prevent this from failing due to limited VRAM
                posterior_batch = {
                    key: tensor.to(torch.device("cpu"))
                    for key, tensor in posterior_batch.items()
                }

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

        # move the posterior samples back to the original device, now that they have been collected
        # posterior_samples = {
        #     key: tensor.to(self.device) for key, tensor in posterior_samples.items()
        # }

        return self.clean_up_posterior_samples(posterior_samples)

    def draw_posterior_samples_from_texts(
        self,
        *texts: str,
        tokens: Sequence[str],
        num_samples: int = 1000,
        return_sites: Optional[Collection[str]] = None,
    ):
        return_sites = return_sites or self.return_sites
        bow_tensor = texts_to_bow_tensor(*texts, tokens=tokens).int()
        ic(bow_tensor)
        ic(torch.arange(len(tokens)).repeat_interleave(bow_tensor[0]))

        bow_tensor = bow_tensor.expand([num_samples * bow_tensor.shape[-2], -1])

        return self.draw_posterior_samples(
            data_loader=sequential_data_loader(
                bow_tensor, device=self.device, dtype=torch.float
            ),
            num_samples=1,
            return_sites=return_sites,
        )

    def calculate_metrics(
        self,
        *data: torch.Tensor,
        targets: torch.Tensor,
        target_site: str,
        batch_size: Optional[int] = None,
        num_samples: int = 1,
        mean_dim: Optional[int] = None,
        cutoff: Optional[float] = None,
        cutoff_compute_method: Literal["grid-search", "base-rate"] = "grid-search",
        post_sample_fun: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        **kwargs,
    ) -> Quality_Result:
        samples = self.draw_posterior_samples(
            data_loader=sequential_data_loader(
                *data,
                batch_size=batch_size,
                device=self.device,
                dtype=torch.float,
            ),
            num_samples=num_samples,
            return_sites=[target_site],
            progress_bar=False,
            **kwargs,
        )[target_site]

        if post_sample_fun is not None:
            samples = post_sample_fun(samples)

        return quality_measures(
            samples,
            targets=targets.to(self.device).float(),
            cutoff=cutoff,
            mean_dim=mean_dim,
        )


class Model(Simple_Model, PyroModule):
    """A Bayesian model that relies on a neural network."""

    def __init__(self, **kwargs) -> None:
        self.svi_pre_hooks = [self.train]
        self.svi_self_pre_hooks = []
        self.svi_step_hooks = []
        self.svi_self_step_hooks = []
        self.svi_post_hooks = [self.eval]
        self.svi_self_post_hooks = []
        super().__init__()

    def draw_posterior_samples(
        self,
        data_loader: Data_Loader,
        num_samples: int = 100,
        parallel_sample: bool = False,
        return_sites: Optional[Collection[str]] = None,
        progress_bar: bool = True,
    ) -> dict[str, torch.Tensor]:
        was_training = self.training
        self.eval()

        with torch.no_grad():
            samples = Simple_Model.draw_posterior_samples(
                self,
                data_loader,
                num_samples,
                parallel_sample,
                return_sites,
                progress_bar,
            )

        if was_training:
            self.train()

        return samples


Post_Sample_Fun = Callable[[torch.Tensor], torch.Tensor]


def eval_samples(
    target_samples: torch.Tensor | Mapping[str, torch.Tensor],
    targets: torch.Tensor | Mapping[str, torch.Tensor],
    target_values: Iterable | Mapping[str, Iterable],
    cutoffs: Optional[
        float | Collection[float] | Mapping[str, None | float | Collection[float]]
    ],
    post_sample_funs: Post_Sample_Fun | Mapping[str, Post_Sample_Fun] | None = None,
) -> dict[str, Quality_Result]:
    if not isinstance(target_samples, Mapping):
        target_samples = {"": target_samples}
    if not isinstance(targets, Mapping):
        targets = {"": targets}
    if not isinstance(target_values, Mapping):
        target_values = {key: target_values for key in targets.keys()}
    if not isinstance(cutoffs, Mapping):
        cutoffs = {key: cutoffs for key in targets.keys()}
    if not isinstance(post_sample_funs, Mapping):
        post_sample_funs = (
            {key: post_sample_funs for key in targets.keys()}
            if post_sample_funs is not None
            else {}
        )

    results = dict()
    for key, target in targets.items():
        samples = target_samples[key]
        if key in post_sample_funs:
            samples = post_sample_funs[key](samples)

        print("------------------------------")
        print(key)

        # global_measures = quality_measures(
        #     samples,
        #     target.to(samples.device).float(),
        #     mean_dim=0,
        #     cutoff=cutoffs[key],
        #     # cutoff=by_discipline.cutoff,
        # )

        # pprint(global_measures)

        by_discipline = quality_measures(
            samples,
            target.to(samples.device).float(),
            mean_dim=0,
            cutoff=cutoffs[key],
            # cutoff=global_measures.cutoff,
            parallel_dim=-1,
        )

        count = target.sum(-2).cpu()

        def weighted_mean(x: torch.Tensor):
            return ((count * x.nan_to_num()).sum() / count.sum()).tolist()

        averaged_measures = Quality_Result(
            accuracy=weighted_mean(torch.tensor(by_discipline.accuracy)),
            precision=weighted_mean(torch.tensor(by_discipline.precision)),
            recall=weighted_mean(torch.tensor(by_discipline.recall)),
            f1_score=weighted_mean(torch.tensor(by_discipline.f1_score)),
            cutoff=by_discipline.cutoff,
        )

        print(f"weighted averages:")
        pprint(
            {
                key: getattr(averaged_measures, key)
                for key in ["accuracy", "precision", "recall", "f1_score"]
            }
        )

        df = pd.DataFrame(
            {
                key: getattr(by_discipline, key)
                for key in ["accuracy", "precision", "recall", "f1_score"]
            }
        )
        df[key] = target_values[key]
        df["support"] = count
        df = df.sort_values(key)
        df = df.set_index(key)
        print(df.sort_values("f1_score", ascending=False).to_string())
        df.to_csv(f"./{key}.csv")
        results[key] = by_discipline

    return results
