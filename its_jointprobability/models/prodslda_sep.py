from __future__ import annotations

import argparse
import math
from collections.abc import Callable, Collection
from pathlib import Path
from typing import Any, Iterable, Optional

import its_jointprobability.models.prodslda as prodslda_module
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pyro
import pyro.distributions as dist
import pyro.infer
import pyro.optim
import torch
import torch.nn.functional as F
from icecream import ic
from its_jointprobability.models.model import Simple_Model, default_data_loader
from its_jointprobability.models.prodslda import ProdSLDA
from its_jointprobability.utils import (
    Quality_Result,
    batch_to_list,
    device,
    get_random_batch_strategy,
    quality_measures,
    texts_to_bow_tensor,
)
from pyro.infer.enum import partial
from tqdm import tqdm

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
        svi_pre_hooks: Optional[Collection[Callable[[], Any]]] = None,
        svi_step_hooks: Optional[Collection[Callable[[], Any]]] = None,
        svi_post_hooks: Optional[Collection[Callable[[], Any]]] = None,
        svi_self_pre_hooks: Optional[Collection[Callable[[Simple_Model], Any]]] = None,
        svi_self_step_hooks: Optional[Collection[Callable[[Simple_Model], Any]]] = None,
        svi_self_post_hooks: Optional[Collection[Callable[[Simple_Model], Any]]] = None,
        **kwargs,
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

        self.svi_pre_hooks = svi_pre_hooks if svi_pre_hooks is not None else []
        self.svi_step_hooks = svi_step_hooks if svi_step_hooks is not None else []
        self.svi_post_hooks = svi_post_hooks if svi_post_hooks is not None else []
        self.svi_self_pre_hooks = (
            svi_self_pre_hooks if svi_self_pre_hooks is not None else []
        )
        self.svi_self_step_hooks = (
            svi_self_step_hooks if svi_self_step_hooks is not None else []
        )
        self.svi_self_post_hooks = (
            svi_self_post_hooks if svi_self_post_hooks is not None else []
        )

        self.accuracies = []

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
    ) -> torch.Tensor:
        labels_plate = pyro.plate("labels", self.label_size, dim=-2)
        docs_plate = pyro.plate("documents-cls", docs.shape[-2], dim=-1)

        with docs_plate:
            logtheta = self.logtheta_prior(docs)
            theta = F.softmax(logtheta, -1)

        # the label application coefficients
        with labels_plate:
            nu = self.nu_prior()

        with docs_plate:
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
    ) -> torch.Tensor:
        # essentially identical to the model,
        # but uses the posterior functions, rather than the prior ones
        labels_plate = pyro.plate("labels", self.label_size, dim=-2)
        docs_plate = pyro.plate("documents-cls", docs.shape[-2], dim=-1)

        with docs_plate:
            logtheta_q = self.logtheta_posterior(docs)
            theta_q = logtheta_q.softmax(-1)

        with labels_plate:
            nu_q = self.nu_posterior()

        with docs_plate:
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
            **kwargs,
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
        batch_size: Optional[int] = None,
        *args,
        **kwargs,
    ):
        """
        Update this model inplace using the given documents and their discipline assignments.
        """
        elbo = pyro.infer.Trace_ELBO(
            num_particles=num_particles, vectorize_particles=True
        )
        data_loader = default_data_loader(docs, labels, batch_size=batch_size)

        # run svi on the given data
        param_store = pyro.get_param_store()
        with param_store.scope() as svi:
            self.run_svi(
                elbo=elbo,
                data_loader=data_loader,
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

    def model_predictive(self, *args, **kwargs) -> pyro.infer.Predictive:
        """
        Only draw samples from the model (not the guide).

        We use this for the Bayesian updates, as they set the model's priors
        to the approximated posteriors and reset the guides.
        Thus, these guides are no longer useful for inference.
        """
        return pyro.infer.Predictive(model=self.model, *args, **kwargs)

    def calculate_accuracy(
        self, docs: torch.Tensor, labels: torch.Tensor, batch_size: Optional[int] = None
    ) -> float:
        return quality_measures(
            self.draw_posterior_samples(
                docs.shape[-2],
                data_args=[docs],
                num_samples=1,
                batch_size=batch_size if batch_size is not None else docs.shape[-2],
                return_sites=["label"],
                progress_bar=False,
            )["label"],
            labels=labels,
            cutoff=1.0,
        ).accuracy  # type: ignore

    def append_to_accuracies_(
        self,
        docs: torch.Tensor,
        labels: torch.Tensor,
        batch_size: Optional[int] = None,
        freq: int = 1,
    ) -> None:
        if len(self.accuracies) % freq == 0:
            self.accuracies.append(self.calculate_accuracy(docs, labels, batch_size))
        else:
            self.accuracies.append(self.accuracies[-1])


def retrain_model(
    path: Path,
    clear_store=True,
    prodslda: Optional[ProdSLDA] = None,
    seed: Optional[int] = None,
    max_epochs: int = 250,
    model_kwargs: Optional[dict[str, Any]] = None,
    **kwargs,
) -> Classification:
    if model_kwargs is None:
        model_kwargs = dict()

    train_data: torch.Tensor = torch.load(
        path / "train_data", map_location=torch.device("cpu")
    ).float()
    train_labels: torch.Tensor = torch.load(
        path / "train_labels", map_location=torch.device("cpu")
    ).float()

    if seed is not None:
        pyro.set_rng_seed(seed)

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
        nu_loc=-4.0,  # prior probability of about 2% to assign a discipline
        nu_scale=5.0,
        a_scale=2.0,
        **model_kwargs,
    )

    batches = batch_to_list(default_data_loader(train_data, train_labels))

    for index, batch in enumerate(batches):
        print(f"epoch {index + 1} / {len(batches)}")
        with pyro.poutine.scale(scale=train_data.shape[-2] / batch[0].shape[-2]):
            model.bayesian_update(
                batch[0],
                batch[1],
                initial_lr=1,
                gamma=0.5,
                num_particles=10,
                max_epochs=max_epochs,
                batch_size=batch[0].shape[-2],
                **kwargs,
            )

    torch.save(model, path / "classification")

    return model


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
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=250,
        help="The maximum number of training epochs per batch of data",
    )
    parser.add_argument("--plot", action="store_true")

    ic.disable()

    args = parser.parse_args()

    train_data: torch.Tensor = torch.load(
        path / "train_data", map_location=device
    ).float()
    train_labels: torch.Tensor = torch.load(
        path / "train_labels", map_location=device
    ).float()

    model = retrain_model(
        Path(args.path),
        seed=args.seed,
        max_epochs=args.max_epochs,
        # if plotting the training process,
        # calculate the training data accuracy after every few epochs
        model_kwargs={
            "svi_self_step_hooks": [
                partial(
                    Classification.append_to_accuracies_,
                    docs=train_data,
                    labels=train_labels.swapaxes(-1, -2),
                    freq=10,
                )
            ]
            if args.plot
            else None
        },
        keep_prev_losses=True,
    )
    prodslda = torch.load(path / "prodslda", map_location=device)

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

    uris: list[str] = torch.load(path / "uris")
    uri_title_dict: dict[str, str] = torch.load(path / "uri_title_dict")
    labels: list[str] = [uri_title_dict[uri] for uri in uris]

    # evaluate the newly trained model on the training data
    print("evaluating model on train data")
    eval_model(model, train_data, train_labels, labels)

    try:
        # evaluate the newly trained model on the testing data
        print("evaluating model on test data")
        test_data: torch.Tensor = torch.load(
            path / "test_data", map_location=device
        ).float()
        test_labels: torch.Tensor = torch.load(
            path / "test_labels", map_location=device
        ).float()

        eval_model(model, test_data, test_labels, labels)

    except FileNotFoundError:
        pass


def eval_model(
    model: Simple_Model,
    data: torch.Tensor,
    labels: torch.Tensor,
    label_values: Iterable,
) -> Quality_Result:
    ic.disable()
    samples = model.draw_posterior_samples(
        data.shape[-2], data_args=[data], return_sites=["a"], num_samples=100
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


def import_data(
    path: Path,
) -> tuple[Classification, dict[int, str], list[str], dict[str, str]]:
    classification = torch.load(path / "classification", map_location=device)
    # ensure that the model is running on the correct device
    # this does not get changed by setting the map location above
    classification.device = device
    dictionary = torch.load(path / "dictionary")
    uris = torch.load(path / "uris")
    uri_title_dict = torch.load(path / "uri_title_dict")

    return classification, dictionary, uris, uri_title_dict


def compare_to_wlo_classification(path: Path):
    import shelve

    import requests

    classification, dictionary, uris, uri_title_dict = import_data(path)
    title_values = [uri_title_dict[uri] for uri in uris]

    test_data: torch.Tensor = torch.load(path / "test_data", map_location=device)
    test_labels: torch.Tensor = torch.load(path / "test_labels", map_location=device)
    train_data: torch.Tensor = torch.load(path / "train_data", map_location=device)
    train_labels: torch.Tensor = torch.load(path / "train_labels", map_location=device)

    comps = []

    for data, labels in zip((test_data, train_data), (test_labels, train_labels)):
        api_url = "http://localhost:8080/predict_subjects"
        wlo_cls = list()
        with shelve.open(str(path / "wlo-classification")) as db:
            for bow in tqdm(data):
                ids = torch.repeat_interleave(bow.int())
                tokens = [dictionary[int(index)] for index in ids]
                text = " ".join(tokens)

                if text in db:
                    wlo_cls.append(db[text])
                    continue

                r = requests.post(api_url, json={"text": " ".join(tokens)})
                if r.ok:
                    prediction = r.json()["disciplines"]
                else:
                    prediction = []

                db[text] = prediction
                wlo_cls.append(prediction)

        wlo_cls_labels = [
            [
                "http://w3id.org/openeduhub/vocabs/discipline/" + value["id"]
                for value in entry
            ]
            for entry in wlo_cls
        ]

        uris_set = set(uris)

        wlo_cls_indices = [
            [uris.index(value) for value in entry if value in uris_set]
            for entry in wlo_cls_labels
        ]

        n = len(uris_set)
        wlo_cls_tensor = torch.stack(
            [
                torch.stack([F.one_hot(torch.tensor(value), n) for value in entry]).sum(
                    0
                )
                if len(entry) > 0
                else torch.zeros(n)
                for entry in wlo_cls_indices
            ]
        )

        qualities_wlo_cls = quality_measures(
            wlo_cls_tensor.unsqueeze(0), labels, cutoff=1, parallel_dim=-1
        )

        samples = classification.draw_posterior_samples(
            len(data), data_args=[data], return_sites=["a"]
        )["a"]

        qualities_new = quality_measures(
            samples,
            labels,
            mean_dim=0,
            parallel_dim=-1,
        )

        def qualitiy_measure_df(quality_measures: Quality_Result) -> pd.DataFrame:
            return (
                pd.DataFrame(
                    {
                        "taxonid": title_values,
                        "accuracy": quality_measures.accuracy,
                        "precision": quality_measures.precision,
                        "recall": quality_measures.recall,
                        "f1-score": quality_measures.f1_score,
                        "count": labels.sum(-2),
                    }
                )
                .set_index("taxonid")
                .fillna(0)
            )

        df_wlo_cls = qualitiy_measure_df(qualities_wlo_cls)
        df_wlo_cls["preds"] = wlo_cls_tensor.sum(-2)
        df_new = qualitiy_measure_df(qualities_new)
        df_new["preds"] = (samples.mean(0) > qualities_new.cutoff).sum(-2)

        joined = df_new.merge(
            df_wlo_cls,
            how="inner",
            left_index=True,
            right_index=True,
            suffixes=(" new", " old"),
        )
        joined["count"] = joined["count new"]
        joined = joined.drop(["count old", "count new"], axis=1)

        for metric in ["accuracy", "recall", "precision", "f1-score"]:
            label_new, label_old = metric + " new", metric + " old"
            joined[metric + " diff"] = joined[label_new] - joined[label_old]

        joined = joined.sort_index(axis=1).sort_values("f1-score diff", ascending=False)

        print(joined)
        comps.append(joined)

    comps[0].to_csv(path / "comparison_test.csv")
    comps[1].to_csv(path / "comparison_train.csv")

    return comps


if __name__ == "__main__":
    retrain_model_cli()
