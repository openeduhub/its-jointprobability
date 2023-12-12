from __future__ import annotations

import argparse
import math
from collections.abc import Collection
from pathlib import Path
from typing import Iterable, Optional

import its_jointprobability.models.prodslda as prodslda_module
import pandas as pd
import pyro
import pyro.distributions as dist
import pyro.infer
import pyro.optim
import torch
import torch.nn.functional as F
from icecream import ic
from its_jointprobability.models.model import Simple_Model
from its_jointprobability.models.prodslda import ProdSLDA
from its_jointprobability.utils import (
    batch_to_list,
    device,
    get_random_batch_strategy,
    quality_measures,
    texts_to_bow_tensor,
)
from pyro.infer.enum import partial

path = Path.cwd() / "data"


class Classification(Simple_Model):
    """
    A fully Bayesian classification model that relies on a given
    topic model (which may not be fully Bayesian) in order assign
    topics to documents.

    These assigned topics are then used linearly for classification.
    """

    return_sites = ("label", "nu", "a")
    return_site_cat_dim = {"nu": 0, "a": -1, "label": -1}

    def __init__(
        self,
        label_size: int,
        prodslda: ProdSLDA,
        observe_negative_labels: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ):
        self.label_size = label_size
        self.observe_negative_labels = (
            observe_negative_labels
            if observe_negative_labels is not None
            else torch.tensor(True, device=device)
        )

        prodslda = prodslda.eval()
        self.num_topics = prodslda.num_topics
        self.prodslda = prodslda
        self.prodlda_logtheta_params = pyro.poutine.block(prodslda.logtheta_params)
        self.prodlda_clean_up_posterior_samples = prodslda.clean_up_posterior_samples
        self.device = device

    def logtheta_prior(
        self,
        docs: torch.Tensor,
        *args,
        logtheta_loc: float = 0.0,
        logtheta_scale: float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        """The prior on the per-document assignment of topics."""
        return pyro.sample(
            "logtheta",
            dist.Normal(
                logtheta_loc * docs.new_ones(self.num_topics),
                logtheta_scale * docs.new_ones(self.num_topics),
            ).to_event(1),
        )

    def logtheta_posterior(self, docs, *args, **kwargs) -> torch.Tensor:
        """
        The variational family on the per-document assignment of topics.

        This directly uses the given topic model in order to obtain the
        hyperparameters for the distribution used here.
        """
        return pyro.sample(
            "logtheta", dist.Normal(*self.prodlda_logtheta_params(docs)).to_event(1)
        )

    def nu_prior(
        self, *args, nu_loc: float = 0.0, nu_scale: float = 10.0, **kwargs
    ) -> torch.Tensor:
        """The prior on the label x topic relevance-matrix."""
        return pyro.sample(
            "nu",
            dist.Normal(
                nu_loc
                * torch.ones(self.label_size, 1, self.num_topics, device=self.device),
                nu_scale
                * torch.ones(self.label_size, 1, self.num_topics, device=self.device),
            ).to_event(1),
        )

    def nu_posterior(
        self,
        *args,
        nu_loc: Optional[torch.Tensor] = None,
        nu_scale: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """The posterior on the label x topic revelance-matrix."""
        nu_loc = (
            nu_loc
            if nu_loc is not None
            else pyro.param(
                "nu_loc",
                lambda: torch.randn(
                    self.label_size, 1, self.num_topics, device=self.device
                ),
            )
        )
        nu_scale = (
            nu_scale
            if nu_scale is not None
            else pyro.param(
                "nu_scale",
                lambda: torch.ones(
                    self.label_size, 1, self.num_topics, device=self.device
                ),
                constraint=dist.constraints.positive,
            )
        )
        return pyro.sample("nu", dist.Normal(nu_loc, nu_scale).to_event(1))

    def a_prior(
        self, nu, theta, *args, a_scale: float = 10.0, **kwargs
    ) -> torch.Tensor:
        """The prior on the applicability of each label, given the topic mix."""
        return pyro.sample(
            "a",
            dist.Normal(
                torch.matmul(nu, theta.swapaxes(-1, -2)).squeeze(-2),
                a_scale,
            ),
        )

    def a_posterior(
        self, nu, theta, *args, a_scale: Optional[torch.Tensor] = None, **kwargs
    ):
        """The posterior on the applicability of each label, given the topic mix."""

        a_scale = (
            a_scale
            if a_scale is not None
            else pyro.param(
                "a_scale",
                lambda: torch.ones(
                    [self.label_size, 1, self.num_topics], device=self.device
                ),
                constraint=dist.constraints.positive,
            )
        )
        return pyro.sample(
            "a",
            dist.Normal(
                torch.matmul(nu, theta.swapaxes(-1, -2)).squeeze(-2),
                torch.matmul(a_scale, theta.swapaxes(-1, -2)).squeeze(-2),
            ),
        )

    def model(
        self,
        docs: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        batch: Optional[Collection[int]] = None,
    ) -> torch.Tensor:
        num_full_data = docs.shape[0]

        if batch is not None:
            batch_size = len(batch)
            docs = docs[list(batch)]
            labels = labels[list(batch)] if labels is not None else None
        else:
            batch_size = num_full_data

        labels_plate = pyro.plate("labels", self.label_size, dim=-2)
        docs_plate = pyro.plate("documents-cls", batch_size, dim=-1)
        scale = pyro.poutine.scale(scale=num_full_data / batch_size)

        with docs_plate, scale:
            logtheta = self.logtheta_prior(docs)
            theta = F.softmax(logtheta, -1)

        # the label application coefficients
        with labels_plate:
            nu = self.nu_prior()

        with docs_plate, scale:
            with labels_plate:
                a = self.a_prior(nu, theta)
                label = pyro.sample(
                    "label",
                    dist.Bernoulli(logits=a),  # type: ignore
                    obs=labels.T if labels is not None else None,
                    # obs_mask=torch.logical_or(
                    #     self.observe_negative_labels, labels.T.bool()
                    # )
                    # if labels is not None
                    # else None,
                    infer={"enumerate": "parallel"},
                )

        return a

    def guide(
        self,
        docs: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        batch: Optional[Collection[int]] = None,
    ) -> torch.Tensor:
        # essentially identical to the model,
        # but uses the posterior functions, rather than the prior ones

        num_full_data = docs.shape[0]

        if batch is not None:
            batch_size = len(batch)
            docs = docs[list(batch)]
            labels = labels[list(batch)] if labels is not None else None
        else:
            batch_size = num_full_data

        labels_plate = pyro.plate("labels", self.label_size, dim=-2)
        docs_plate = pyro.plate("documents-cls", batch_size, dim=-1)
        scale = pyro.poutine.scale(scale=num_full_data / batch_size)

        with docs_plate, scale:
            logtheta_q = self.logtheta_posterior(docs)
            theta_q = logtheta_q.softmax(-1)

        with labels_plate:
            nu_q = self.nu_posterior()

        with docs_plate, scale:
            with labels_plate:
                return self.a_posterior(nu_q, theta_q)

    def clean_up_posterior_samples(
        self, posterior_samples: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
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
        """Helper functions in order to draw posterior samples for texts."""
        return_sites = return_sites if return_sites is not None else self.return_sites
        bow_tensor = texts_to_bow_tensor(*texts, token_dict=token_dict)
        return self.draw_posterior_samples(
            data_len=bow_tensor.shape[0],
            data_args=[bow_tensor],
            num_samples=num_samples,
            return_sites=return_sites,
        )

    @classmethod
    def with_priors(
        cls,
        label_size: int,
        prodslda: ProdSLDA,
        observe_negative_labels: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> Classification:
        """
        Create a model with some specified priors.

        The priors may be specified through the keyword arguments,
        which are passed to each prior function.
        """
        obj = cls(
            label_size=label_size,
            prodslda=prodslda,
            observe_negative_labels=observe_negative_labels,
            device=device,
        )
        obj.nu_prior = partial(obj.nu_prior, **kwargs)
        obj.logtheta_prior = partial(obj.logtheta_prior, **kwargs)
        obj.a_prior = partial(obj.a_prior, **kwargs)
        return obj

    def bayesian_update(
        self,
        docs: torch.Tensor,
        labels: torch.Tensor,
        num_particles=3,
        *args,
        **kwargs,
    ) -> list[float]:
        """
        Update this model inplace using the given documents and their discipline assignments.
        """
        elbo = pyro.infer.Trace_ELBO(
            num_particles=num_particles, vectorize_particles=True
        )

        # run svi on the given data
        param_store = pyro.get_param_store()
        with param_store.scope() as svi:
            losses = self.run_svi(
                elbo=elbo,
                train_data_len=docs.shape[-2],
                train_args=[docs, labels],
                *args,
                **kwargs,
            )

        with param_store.scope(svi):
            nu_loc = pyro.param("nu_loc").detach()
            nu_scale = pyro.param("nu_scale").detach()
            a_scale = pyro.param("a_scale").detach()

        self.logtheta_prior = self.logtheta_posterior
        self.nu_prior = partial(self.nu_posterior, nu_loc=nu_loc, nu_scale=nu_scale)  # type: ignore
        self.a_prior = partial(self.a_posterior, a_scale=a_scale)  # type: ignore

        # no longer use the posterior distribution in the predictive
        self.predictive = self.model_predictive

        return losses

    def model_predictive(self, *args, **kwargs) -> pyro.infer.Predictive:
        """
        Only draw samples from the model (not the guide).

        We use this for the Bayesian updates, as they set the model's priors
        to the approximated posteriors and reset the guides.
        Thus, these guides are no longer useful for inference.
        """
        return pyro.infer.Predictive(model=self.model, *args, **kwargs)


def retrain_model(
    path: Path, clear_store=True, prodslda: Optional[ProdSLDA] = None
) -> Classification:
    train_data: torch.Tensor = torch.load(path / "train_data", map_location=device)
    train_labels: torch.Tensor = torch.load(path / "train_labels", map_location=device)

    if clear_store:
        pyro.get_param_store().clear()

    if prodslda is None:
        try:
            prodslda = torch.load(path / "prodslda", map_location=device)
            # if the model is None, the import was clearly not successful
            if prodslda is None:
                raise FileNotFoundError

        except FileNotFoundError:
            # if the topic model is missing, generate it
            print("training topic model")
            prodslda = prodslda_module.retrain_model(path)

    print("training classification")
    model = Classification.with_priors(
        label_size=train_labels.shape[-1],
        prodslda=prodslda,
        observe_negative_labels=torch.tensor(True, device=device),
        device=device,
        nu_loc=0.0,
        nu_scale=10.0,
    )

    pyro.set_rng_seed(1)
    num_epochs = 5
    batch_size = math.ceil(train_data.shape[-2] / num_epochs)
    batch_strategy = get_random_batch_strategy(train_data.shape[-2], batch_size)

    for index, batch_ids in enumerate(batch_to_list(batch_strategy)):
        print(f"epoch {index + 1} / {num_epochs}")
        docs_batch = train_data[..., batch_ids, :]
        labels_batch = train_labels[..., batch_ids, :]
        with pyro.poutine.scale(scale=train_data.shape[-2] / docs_batch.shape[-2]):
            model.bayesian_update(
                docs_batch,
                labels_batch,
                initial_lr=1,
                gamma=0.001,
                num_particles=8,
                max_epochs=250,
                batch_size=docs_batch.shape[-2],
            )

    torch.save(model, path / "classification")

    return model


def retrain_model_cli():
    """Add some CLI arguments to the retraining of the model."""
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)

    ic.disable()

    args = parser.parse_args()
    model = retrain_model(Path(args.path))

    labels: torch.Tensor = torch.load(path / "labels")

    # evaluate the newly trained model on the training data
    print("evaluating model on train data")
    train_data: torch.Tensor = torch.load(path / "train_data", map_location=device)
    train_labels: torch.Tensor = torch.load(path / "train_labels", map_location=device)
    eval_model(model, train_data, train_labels, labels)

    try:
        # evaluate the newly trained model on the testing data
        print("evaluating model on test data")
        test_data: torch.Tensor = torch.load(path / "test_data", map_location=device)
        test_labels: torch.Tensor = torch.load(
            path / "test_labels", map_location=device
        )

        eval_model(model, test_data, test_labels, labels)
    except FileNotFoundError:
        pass


def eval_model(
    model: Classification,
    data: torch.Tensor,
    labels: torch.Tensor,
    label_values: Iterable,
):
    ic.disable()
    samples = model.draw_posterior_samples(
        data.shape[-2], data_args=[data], return_sites=["a"], num_samples=100
    )["a"]
    global_measures = quality_measures(samples, labels, mean_dim=0, cutoff=None)
    print(f"global measures: {global_measures}")

    by_discipline = quality_measures(
        samples,
        labels,
        mean_dim=0,
        cutoff=global_measures.cutoff,
        parallel_dim=-1,
    )
    df = pd.DataFrame(
        {
            key: getattr(by_discipline, key).cpu()
            for key in ["accuracy", "precision", "recall", "f1_score"]
        }
    )
    df["taxonid"] = label_values
    df["count"] = labels.sum(-2).cpu()
    df = df.set_index("taxonid")
    print(df.sort_values("f1_score", ascending=False))


def import_data(
    path: Path,
) -> tuple[Classification, dict[int, str], list[str], list[str]]:
    classification = torch.load(path / "classification", map_location=device)
    # ensure that the model is running on the correct device
    # this does not get changed by setting the map location above
    classification.device = device
    dictionary = torch.load(path / "dictionary")
    labels = torch.load(path / "labels")
    uris = torch.load(path / "uris")

    return classification, dictionary, labels, uris


if __name__ == "__main__":
    retrain_model_cli()
