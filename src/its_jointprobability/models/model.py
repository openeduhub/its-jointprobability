from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from pprint import pprint
from typing import Any, Optional

import numpy as np
import pandas as pd
import pyro
import pyro.infer
import pyro.ops.stats
import pyro.optim
import torch
from its_jointprobability.utils import (
    Data_Loader,
    Quality_Result,
    batch_to_list,
    get_batch_size,
    quality_measures,
    sequential_data_loader,
    texts_to_bow_tensor,
)
from pydantic import BaseModel, Field
from pyro.nn.module import PyroModule
from tqdm import tqdm, trange


class Prediction_Score(BaseModel):
    """An individual prediction for a particular target."""

    id: str
    name: str
    mean_prob: float
    median_prob: float
    baseline_diff: float
    prob_interval: list[float] = Field(..., examples=[[0.1, 0.5]])


def iterate_over_independent_samples(
    samples: dict[str, torch.Tensor], dim: int = 0
) -> Iterator[dict[str, torch.Tensor]]:
    """Iterate over the left-most dimension of the given per-field samples."""
    # swap dimensions such that the dimension to iterate over is left-most
    samples = {key: value.swapaxes(0, dim) for key, value in samples.items()}
    for sample in zip(*samples.values()):
        sample = list(sample)
        yield dict(zip(samples.keys(), sample))


class Simple_Model:
    """Base class for Bayesian models that are learned through SVI."""

    # the random variables that are contained within posterior samples
    return_sites: Collection[str]
    # the dimensions along which to concat the random variables during batching
    return_site_cat_dim: Mapping[str, int]
    # map from target name to site used for prediction
    prediction_sites: Mapping[str, str]

    # the baseline, "empty" data
    baseline_data: Collection[torch.Tensor]

    # the names of the various targets being predicted
    target_names: Sequence[str]
    # the number of categories per target being predicted
    target_sizes: Sequence[int]
    # per-target mapping from category URI to human-readable label
    id_label_dicts: Sequence[Mapping[str, str]]

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

    # an annealing factor that increases to 1.0 during training.
    # it is up to the actual model implementation to use this factor.
    annealing_factor: float = 1.0

    # the cached baseline distribution of each target's categories
    _baselines: Optional[dict[str, list[float]]] = None

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

        # reset the cached baseline distribution
        self._baselines = None

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
        # ensure that the return sites are a tuple, as Predictive requires this
        # to be a list, tuple, or set
        if "return_sites" in kwargs:
            kwargs["return_sites"] = tuple(kwargs["return_sites"])
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
            num_samples=num_samples,
            return_sites=return_sites,
            parallel=parallel_sample,
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
    ) -> dict[str, torch.Tensor]:
        return_sites = return_sites or self.return_sites

        # because we usually use significantly higher numbers of samples here,
        # use the fact that each document is assumed to be independent and
        # simply represent the number of samples as a duplication of the
        # inputs. this allows use to utilize the usual batching methodology
        bow_tensor = (
            texts_to_bow_tensor(*texts, tokens=tokens)
            .int()
            .expand([num_samples, -1, -1])
            .reshape([num_samples * len(texts), -1])
        )

        posterior_samples = self.draw_posterior_samples(
            data_loader=sequential_data_loader(
                bow_tensor, device=self.device, dtype=torch.float
            ),
            num_samples=1,
            return_sites=return_sites,
            parallel_sample=True,
        )

        # convert the flattened shape=[batch * docs, target] samples back into
        # their usual [batch, docs, target] shape
        posterior_samples = {
            key: value.reshape([num_samples, len(texts), -1])
            for key, value in posterior_samples.items()
        }

        return posterior_samples

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

    def get_baseline_dists(self) -> dict[str, list[float]]:
        if self._baselines is not None:
            return self._baselines

        baseline_samples = self.draw_posterior_samples(
            sequential_data_loader(
                *[
                    baseline_date.expand([10**4] + [-1 for _ in baseline_date.shape])
                    for baseline_date in self.baseline_data
                ]
            ),
            num_samples=1,
            return_sites=self.prediction_sites.values(),
            progress_bar=True,
        )

        # remove the leading empty dimension from the samples
        baseline_samples = {
            key: value.squeeze(0) for key, value in baseline_samples.items()
        }

        prediction_scores = next(
            self._predict_from_posterior_samples(
                iterate_over_independent_samples(baseline_samples),
                interval_size=0.0,
                # avoid infinite recursion
                compute_baselines=False,
            )
        )

        self._baselines = {
            key: [prediction.mean_prob for prediction in predictions]
            for key, predictions in prediction_scores.items()
        }
        return self._baselines

    def _predict_from_posterior_samples(
        self,
        posterior_samples_by_text: Iterable[Mapping[str, torch.Tensor]],
        interval_size: float,
        compute_baselines: bool = True,
    ) -> Iterator[dict[str, list[Prediction_Score]]]:
        for posterior_samples_by_field in posterior_samples_by_text:
            predictions: dict[str, list[Prediction_Score]] = dict()

            for field, posterior_samples, id_label_dict in zip(
                self.target_names,
                posterior_samples_by_field.values(),
                self.id_label_dicts,
            ):
                probs = posterior_samples
                mean_probs = probs.mean(0)
                median_probs = probs.median(0)[0]
                intervals: list[list[float]] = [
                    interval.tolist()
                    for interval in (
                        pyro.ops.stats.hpdi(probs, interval_size).squeeze(-1).T
                    )
                ]

                mean_prob_diffs = torch.zeros_like(mean_probs)
                if compute_baselines:
                    mean_prob_diffs = mean_probs - torch.tensor(
                        self.get_baseline_dists()[field]
                    )

                prediction = [
                    Prediction_Score(
                        id=uri,
                        name=label,
                        mean_prob=float(mean_prob),
                        baseline_diff=mean_prob_diff,
                        median_prob=float(median_prob),
                        prob_interval=interval,
                    )
                    for label, uri, mean_prob, mean_prob_diff, median_prob, interval in zip(
                        id_label_dict.values(),
                        id_label_dict.keys(),
                        mean_probs,
                        mean_prob_diffs,
                        median_probs,
                        intervals,
                    )
                ]
                predictions[field] = prediction

            yield predictions

    def predict_from_texts(
        self,
        *texts: str,
        tokens: Sequence[str],
        num_samples: int = 100,
        interval_size: float = 0.8,
    ) -> Iterator[dict[str, list[Prediction_Score]]]:
        posterior_samples = self.draw_posterior_samples_from_texts(
            *texts,
            tokens=tokens,
            num_samples=num_samples,
            return_sites=self.prediction_sites.values(),
        )

        return self._predict_from_posterior_samples(
            iterate_over_independent_samples(posterior_samples, dim=-2),
            interval_size=interval_size,
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
