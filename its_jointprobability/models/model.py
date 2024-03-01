from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Collection, Iterable, Sequence
from functools import partial
from typing import Any, Optional, TypeVar

import numpy as np
import optuna
import pandas as pd
import pyro
import pyro.infer
import pyro.optim
import torch
import torch.nn.functional as F
from icecream import ic
from its_jointprobability.utils import (
    Batch,
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

    # an annealing factor that rises during training
    annealing_factor: float

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
        # so that we can correctly scale the ELBO of each mini-batch
        # and correct the learning rate decay
        one_epoch = batch_to_list(data_loader)
        batches_per_epoch = len(one_epoch)
        n = 0
        for batch in one_epoch:
            n += get_batch_size(batch)

        assert n > 0

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
            batch_losses = list()
            for last_batch_in_epoch, batch in data_loader:
                batch_size = get_batch_size(batch)
                with pyro.poutine.scale(scale=n / batch_size):
                    batch_losses.append(svi.step(*batch))
                # break if this was the last batch
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
        posterior_samples = {
            key: tensor.to(self.device)
            for key, tensor in posterior_samples.items()
        }

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
        return self.draw_posterior_samples(
            data_loader=sequential_data_loader(
                bow_tensor, device=self.device, dtype=torch.float
            ),
            num_samples=num_samples,
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
        post_sample_fun: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
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


T = TypeVar("T", bound=Model)


def prune(
    obj: T,
    epoch: int,
    loss: float,
    eval_fun: Callable[[T, int, float], float],
    trial: optuna.trial.Trial,
    min_epochs: int = 0,
    freq: int = 1,
):
    if epoch % freq == 0:
        val = eval_fun(obj, epoch, loss)
        trial.report(val, epoch)
        if trial.should_prune() and epoch >= min_epochs:
            ic(epoch, val)
            raise optuna.TrialPruned()


def set_up_optuna_study(
    factory: Callable[..., T],
    data_loader: Data_Loader,
    elbo_choices: dict[str, type[pyro.infer.ELBO]],
    eval_funs_final: Collection[Callable[[T], float]],
    eval_fun_prune: Optional[Callable[[T, int, float], float]] = None,
    eval_freq: int = 10,
    var_model_kwargs: Optional[dict[str, Callable[[optuna.trial.Trial], Any]]] = None,
    fix_model_kwargs: Optional[dict[str, Any]] = None,
    initial_lr: Optional[Callable[[optuna.trial.Trial], float]] = None,
    gamma: Optional[Callable[[optuna.trial.Trial], float]] = None,
    num_particles: Optional[Callable[[optuna.trial.Trial], int]] = None,
    max_epochs: Optional[Callable[[optuna.trial.Trial], int]] = None,
    min_epochs=10,
    vectorize_particles: bool = True,
    seed: Optional[int] = 42,
    device: Optional[torch.device] = None,
    hooks: Optional[dict[str, Collection[Callable]]] = None,
) -> Callable[[optuna.trial.Trial], list[float]]:
    if var_model_kwargs is None:
        var_model_kwargs = dict()
    if fix_model_kwargs is None:
        fix_model_kwargs = dict()
    if hooks is None:
        hooks = dict()

    if initial_lr is None:
        initial_lr = lambda trial: trial.suggest_float(
            "initial_lr", 1e-5, 1e-1, log=True
        )
    if gamma is None:
        gamma = lambda trial: trial.suggest_float("gamma", 0.01, 1.0, log=True)
    if num_particles is None:
        num_particles = lambda trial: trial.suggest_int("num_particles", 1, 5)
    if max_epochs is None:
        max_epochs = lambda trial: trial.suggest_int(
            "max_epoches", min_epochs, min_epochs + 50 * 6, 50
        )

    def objective(trial: optuna.trial.Trial) -> list[float]:
        trial_kwargs = {key: fun(trial) for key, fun in var_model_kwargs.items()}
        initial_lr_trial = initial_lr(trial)
        gamma_trial = gamma(trial)
        num_particles_trial = num_particles(trial)
        trial_elbo_num = trial.suggest_categorical(
            "elbo_num", list(elbo_choices.keys())
        )
        # adjust the maximum number of epochs w.r.t. the number of particles
        # to keep training time roughly constant
        max_epochs_trial = int(max_epochs(trial) / num_particles_trial)

        pyro.get_param_store().clear()
        # to prevent the experiment from studying different
        # initializations accidentally
        pyro.set_rng_seed(seed)

        model = factory(device=device, **(fix_model_kwargs | trial_kwargs))

        for hook_type, hook_funs in hooks.items():
            for hook_fun in hook_funs:
                getattr(model, hook_type).append(hook_fun)

        if eval_fun_prune is not None:
            model.svi_self_step_hooks.append(
                partial(
                    prune,
                    eval_fun=eval_fun_prune,
                    trial=trial,
                    min_epochs=min_epochs,
                    freq=max(1, int(eval_freq / num_particles_trial)),
                )
            )

        trial_elbo = elbo_choices[trial_elbo_num](
            num_particles=num_particles_trial, vectorize_particles=vectorize_particles
        )

        model.run_svi(
            elbo=trial_elbo,
            data_loader=data_loader,
            initial_lr=initial_lr_trial,
            gamma=gamma_trial,
            min_epochs=min_epochs,
            max_epochs=max_epochs_trial,
            metric_len=10,
            min_z_score=1.68,
        )

        return [fun(model) for fun in eval_funs_final]

    return objective


Post_Sample_Fun = Callable[[torch.Tensor], torch.Tensor]


def eval_model(
    model: Simple_Model,
    *data: torch.Tensor | None,
    targets: torch.Tensor | dict[str, torch.Tensor],
    target_values: Iterable | dict[str, Iterable],
    eval_sites: str | dict[str, str],
    cutoffs: Optional[float] | dict[str, Optional[float]],
    post_sample_funs: Post_Sample_Fun | dict[str, Post_Sample_Fun] | None = None,
) -> dict[str, Quality_Result]:
    if not isinstance(targets, dict):
        targets = {"": targets}
    if not isinstance(target_values, dict):
        target_values = {key: target_values for key in targets.keys()}
    if not isinstance(eval_sites, dict):
        eval_sites = {key: eval_sites for key in targets.keys()}
    if not isinstance(cutoffs, dict):
        cutoffs = {key: cutoffs for key in targets.keys()}
    if not isinstance(post_sample_funs, dict):
        post_sample_funs = (
            {key: post_sample_funs for key in targets.keys()}
            if post_sample_funs is not None
            else {}
        )

    samples_by_key = model.draw_posterior_samples(
        data_loader=sequential_data_loader(
            *data,
            device=model.device,
            batch_size=1000,
            dtype=torch.float,
        ),
        return_sites=list(eval_sites.values()),
        num_samples=100,
    )

    results = dict()
    for (key, target), samples in zip(targets.items(), samples_by_key.values()):
        if key in post_sample_funs:
            samples = post_sample_funs[key](samples)

        print("------------------------------")
        print(key)
        global_measures = quality_measures(
            samples, target.to(model.device).float(), mean_dim=0, cutoff=cutoffs[key]
        )
        print(f"global measures: {global_measures}")

        by_discipline = quality_measures(
            samples,
            target.to(model.device).float(),
            mean_dim=0,
            cutoff=global_measures.cutoff,
            parallel_dim=-1,
        )
        df = pd.DataFrame(
            {
                key: getattr(by_discipline, key)
                for key in ["accuracy", "precision", "recall", "f1_score"]
            }
        )
        df["taxonid"] = target_values[key]
        df["count"] = target.sum(-2).cpu()
        df = df.set_index("taxonid")
        print(df.sort_values("f1_score", ascending=False))

        results[key] = by_discipline

    return results
