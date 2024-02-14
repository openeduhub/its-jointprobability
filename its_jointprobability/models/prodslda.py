import argparse
import math
import operator as op
from collections.abc import Collection, Sequence
from functools import reduce
from pathlib import Path
from typing import NamedTuple, Optional

import its_jointprobability.models.prodslda as prodslda_module
import optuna
import pyro
import pyro.distributions as dist
import pyro.infer
import pyro.optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic
from its_jointprobability.data import (
    Split_Data,
    load_data,
    make_data,
    save_data,
    save_model,
)
from its_jointprobability.models.model import Model, eval_model, set_up_optuna_study
from its_jointprobability.utils import Data_Loader, default_data_loader


class ProdSLDA(Model):
    """
    A modification of the ProdLDA model to support supervised classification.
    """

    def __init__(
        self,
        vocab: Sequence[str],
        id_label_dicts: Collection[dict[str, str]],
        target_names: Collection[str],
        num_topics: int,
        cov_rank: Optional[int] = None,
        hid_size: int = 100,
        hid_num: int = 1,
        dropout: float = 0.2,
        nu_loc: float = 0.0,
        nu_scale: float = 10.0,
        target_scale: float = 1.0,
        use_batch_normalization: bool = True,
        observe_negative_targets: torch.Tensor = torch.tensor(True),
        device: Optional[torch.device] = None,
    ):
        vocab_size = len(vocab)
        target_sizes = [len(id_label_dict) for id_label_dict in id_label_dicts]

        self.return_sites = tuple(
            reduce(
                op.add,
                [[f"target_{i}"] for i, _ in enumerate(target_sizes)] + [["nu", "a"]],
            )
        )
        self.return_site_cat_dim = reduce(
            op.or_,
            [
                {
                    f"target_{i}": -2,
                }
                for i, _ in enumerate(target_sizes)
            ]
            + [{"nu": -4, "a": -2}],
        )

        super().__init__()
        self.vocab = vocab
        self.vocab_size = vocab_size
        self.id_label_dicts = id_label_dicts
        self.target_names = target_names
        self.target_sizes = target_sizes
        self.num_topics = num_topics
        self.cov_rank = (
            cov_rank
            if cov_rank is not None
            else math.ceil(math.sqrt(sum(target_sizes)))
        )
        self.hid_size = hid_size
        self.hid_num = hid_num
        self.dropout = dropout
        self.nu_loc = nu_loc
        self.nu_scale = nu_scale
        self.target_scale = target_scale
        self.use_batch_normalization = use_batch_normalization
        self.observe_negative_targets = observe_negative_targets
        self.device = device

        self.args = {
            "vocab": self.vocab,
            "id_label_dicts": self.id_label_dicts,
            "target_names": self.target_names,
            "num_topics": self.num_topics,
            "cov_rank": self.cov_rank,
            "hid_size": self.hid_size,
            "hid_num": self.hid_num,
            "dropout": self.dropout,
            "nu_loc": self.nu_loc,
            "nu_scale": self.nu_scale,
            "target_scale": self.target_scale,
            "use_batch_normalization": self.use_batch_normalization,
            "observe_negative_targets": self.observe_negative_targets,
        }

        self.decoder = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_topics, vocab_size),
        )
        if use_batch_normalization:
            self.decoder.append(nn.BatchNorm1d(vocab_size, affine=False))
        self.decoder.append(nn.Softmax(-1))

        self.encoder = nn.Sequential(
            nn.Linear(vocab_size, hid_size),
        )

        for _ in range(hid_num - 1):
            self.encoder.append(nn.Sigmoid())
            self.encoder.append(nn.Linear(hid_size, hid_size))

        self.encoder.append(nn.Sigmoid())
        self.encoder.append(nn.Dropout(dropout))
        self.encoder.append(nn.Linear(hid_size, num_topics * 2))
        if use_batch_normalization:
            self.encoder.append(nn.BatchNorm1d(num_topics * 2, affine=False))

        self.to(device)

    def model(
        self,
        docs: torch.Tensor,
        *targets: torch.Tensor,
        obs_masks: Optional[Sequence[torch.Tensor]] = None,
    ):
        # if no observations mask has been given, ignore any docs that
        # do not have any assigned labels for that given target
        if obs_masks is None:
            obs_masks = [target.sum(-1) > 0 for target in targets]

        docs_plate = pyro.plate("documents_plate", docs.shape[0], dim=-1)

        nu = pyro.sample(
            "nu",
            dist.Normal(
                torch.tensor(self.nu_loc, device=docs.device),
                torch.tensor(self.nu_scale, device=docs.device),
            )
            .expand(torch.Size([self.num_topics, sum(self.target_sizes)]))
            .to_event(2),
        )

        ic(nu.shape)

        with docs_plate:
            logtheta_loc = docs.new_zeros(self.num_topics)
            logtheta_scale = docs.new_ones(self.num_topics)
            logtheta = pyro.sample(
                "logtheta", dist.Normal(logtheta_loc, logtheta_scale).to_event(1)
            )
            if len(logtheta.shape) > 2:
                logtheta = logtheta.squeeze(0)
            theta = F.softmax(logtheta, -1)

            ic(theta.shape)

        pyro.module("decoder", self.decoder, update_module_params=True)
        count_param = self.decoder(theta)

        with docs_plate:
            # Currently, PyTorch Multinomial requires `total_count` to be homogeneous.
            # Because the numbers of words across documents can vary,
            # we will use the maximum count accross documents here.
            # This does not affect the result because Multinomial.log_prob does
            # not require `total_count` to evaluate the log probability.
            total_count = int(docs.sum(-1).max())
            pyro.sample(
                "obs",
                dist.Multinomial(total_count, count_param),
                obs=docs,
            )

            a = pyro.sample(
                "a",
                dist.Normal(torch.matmul(theta, nu), 1).to_event(1),
            )
            ic(a.shape)
            a_split = a.split(self.target_sizes, -1)

            with pyro.poutine.scale(scale=self.target_scale):
                for i in range(len(self.target_sizes)):
                    target = pyro.sample(
                        f"target_{i}",
                        dist.Bernoulli(logits=a_split[i]).to_event(1),  # type: ignore
                        obs=targets[i] if len(targets) > i else None,
                        obs_mask=obs_masks[i] if len(obs_masks) > i else None,
                    )
                    ic(target.shape)

    def logtheta_params(self, doc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pyro.module("encoder", self.encoder, update_module_params=True)
        logtheta_loc, logtheta_logvar = self.encoder(doc).split(self.num_topics, -1)
        logtheta_scale = F.softplus(logtheta_logvar) + 1e-7

        return logtheta_loc, logtheta_scale

    def guide(
        self,
        docs: torch.Tensor,
        *targets: torch.Tensor,
        obs_masks: Optional[Sequence[torch.Tensor]] = None,
    ):
        n = sum(self.target_sizes)
        if obs_masks is None:
            obs_masks = [target.sum(-1) > 0 for target in targets]

        docs_plate = pyro.plate("documents_plate", docs.shape[0], dim=-1)

        # variational parameters for the relationship between topics and targets
        mu_q = pyro.param(
            "mu", lambda: torch.randn(self.num_topics, n, device=docs.device)
        )
        cov_factor = pyro.param(
            "cov_factor",
            lambda: docs.new_ones(self.num_topics, n, self.cov_rank),
            constraint=dist.constraints.positive,
        )
        cov_diag = pyro.param(
            "cov_diag",
            lambda: docs.new_ones(self.num_topics, n),
            constraint=dist.constraints.positive,
        )

        ic(mu_q.shape)
        ic(cov_diag.shape)

        nu_q = pyro.sample(
            "nu",
            dist.LowRankMultivariateNormal(mu_q, cov_factor, cov_diag).to_event(1),
        )

        ic(nu_q.shape)

        with docs_plate:
            logtheta_q = pyro.sample(
                "logtheta", dist.Normal(*self.logtheta_params(docs)).to_event(1)
            )
            theta_q = F.softmax(logtheta_q, -1)

            a_q_scale = pyro.param(
                "a_q_scale",
                lambda: docs.new_ones(n),
                constraint=dist.constraints.positive,
            )

            a_q = pyro.sample(
                "a",
                dist.Normal(torch.matmul(theta_q, nu_q), a_q_scale).to_event(1),
            )
            ic(a_q.shape)
            a_q_split = a_q.split(self.target_sizes, -1)

            with pyro.poutine.scale(scale=self.target_scale):
                for i in range(len(self.target_sizes)):
                    target = pyro.sample(
                        f"target_{i}_unobserved",
                        dist.Bernoulli(logits=a_q_split[i]).to_event(1),  # type: ignore
                    )

    def clean_up_posterior_samples(
        self, posterior_samples: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        for i, _ in enumerate(self.target_sizes):
            target = f"target_{i}"
            if target in posterior_samples:
                ic(posterior_samples[target].shape)

        a = "a"
        nu = "nu"
        if a in posterior_samples:
            ic(posterior_samples[a].shape)
        if nu in posterior_samples:
            posterior_samples[nu] = posterior_samples[nu].squeeze(-3)
            ic(posterior_samples[nu].shape)

        return posterior_samples

    def beta(self):
        # beta matrix elements are the weights of the FC layer on the decoder
        weights = self.decoder.beta.weight.T
        return F.softmax(weights, dim=-1).detach()


class Data(NamedTuple):
    train_docs: torch.Tensor
    train_targets: dict[str, torch.Tensor]
    test_docs: torch.Tensor
    test_targets: dict[str, torch.Tensor]


def set_up_data(data: Split_Data) -> Data:
    train_data = data.train
    test_data = data.test

    train_docs: torch.Tensor = torch.tensor(train_data.bows)
    train_targets: dict[str, torch.Tensor] = {
        key: torch.tensor(value.arr) for key, value in train_data.target_data.items()
    }

    ic(train_docs.shape)
    ic({field: train_target.shape for field, train_target in train_targets.items()})
    ic({field: train_target.sum(-2) for field, train_target in train_targets.items()})

    test_docs: torch.Tensor = torch.tensor(test_data.bows)
    test_targets: dict[str, torch.Tensor] = {
        key: torch.tensor(value.arr) for key, value in test_data.target_data.items()
    }

    ic(test_docs.shape)
    ic({field: test_target.shape for field, test_target in test_targets.items()})
    ic({field: test_target.sum(-2) for field, test_target in test_targets.items()})

    return Data(
        train_docs=train_docs,
        train_targets=train_targets,
        test_docs=test_docs,
        test_targets=test_targets,
    )


def train_model(
    data_loader: Data_Loader,
    vocab: Sequence[str],
    id_label_dicts: Collection[dict[str, str]],
    target_names: Collection[str],
    num_topics: int = 50,
    hid_size: int = 500,
    hid_num: int = 1,
    dropout: float = 0.2,
    nu_loc: float = -1.7,
    nu_scale: float = 15,
    min_epochs: int = 100,
    max_epochs: int = 500,
    target_scale: float = 1.0,
    initial_lr: float = 0.1,
    gamma: float = 0.75,
    device: Optional[torch.device] = None,
    seed: int = 0,
) -> ProdSLDA:
    pyro.set_rng_seed(seed)

    prodslda = ProdSLDA(
        vocab=vocab,
        id_label_dicts=id_label_dicts,
        target_names=target_names,
        num_topics=num_topics,
        hid_size=hid_size,
        hid_num=hid_num,
        dropout=dropout,
        nu_loc=nu_loc,
        nu_scale=nu_scale,
        target_scale=target_scale,
        observe_negative_targets=torch.tensor(True, device=device),
        device=device,
    )

    prodslda.run_svi(
        data_loader=data_loader,
        elbo=pyro.infer.TraceGraph_ELBO(
            num_particles=3,
            vectorize_particles=False,
        ),
        min_epochs=min_epochs,
        max_epochs=max_epochs,
        initial_lr=initial_lr,
        gamma=gamma,
    )

    return prodslda.eval()


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
    parser.add_argument(
        "--max-len",
        type=int,
        default=None,
        help="The maximum number of training documents",
    )
    parser.add_argument(
        "--skip-cache",
        action="store_true",
        help="Whether to skip the cached train / test data, effectively forcing a re-generation.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether to print various logs to the stdout during training.",
    )

    args = parser.parse_args()
    ic.enabled = args.verbose

    path = Path(args.path)

    try:
        if args.skip_cache:
            raise FileNotFoundError()
        data = load_data(path)
    except FileNotFoundError:
        print("Processed data not found. Generating it...")
        data = make_data(path, always_include_confirmed=True, max_len=args.max_len)
        save_data(path, data)

    train_docs, train_targets, test_docs, test_targets = set_up_data(data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_loader = default_data_loader(
        train_docs, *train_targets.values(), device=device, dtype=torch.float
    )

    prodslda = train_model(
        data_loader,
        vocab=data.train.words.tolist(),
        id_label_dicts=[
            {
                id: label
                for id, label in zip(
                    data.train.target_data[field].uris,
                    data.train.target_data[field].labels,
                )
            }
            for field in train_targets.keys()
        ],
        target_names=list(data.train.target_data.keys()),
        device=device,
        seed=args.seed,
        max_epochs=args.max_epochs,
    )

    save_model(prodslda, path)

    titles: list[str] = reduce(
        op.add, (field.labels.tolist() for field in data.train.target_data.values()), []
    )

    # evaluate the newly trained model
    print("evaluating model on train data")
    eval_model(
        prodslda,
        train_docs,
        torch.concat(list(train_targets.values()), -1),
        titles,
        eval_site=f"a",
    )

    if len(test_docs) > 0:
        print("evaluating model on test data")
        eval_model(
            prodslda,
            test_docs,
            torch.concat(list(test_targets.values()), -1),
            titles,
            eval_site=f"a",
        )


def run_optuna_study(
    path: Path,
    n_trials=25,
    seed: int = 0,
    device: Optional[torch.device] = None,
):
    data = load_data(path)
    train_docs, train_targets, test_docs, test_targets = set_up_data(data)

    objective = set_up_optuna_study(
        factory=ProdSLDA,
        data_loader=default_data_loader(
            train_docs,
            *train_targets.values(),
            device=device,
            dtype=torch.float,
        ),
        elbo_choices={
            "TraceGraph": pyro.infer.TraceGraph_ELBO,
        },
        eval_funs_final=[
            lambda obj: obj.calculate_metrics(
                test_docs,
                targets=test_targets,
                target_site="target",
                num_samples=100,
                mean_dim=0,
                cutoff=0.2,
            ).accuracy,  # type: ignore
        ],
        eval_fun_prune=lambda obj, epoch, loss: obj.calculate_metrics(
            test_docs,
            targets=test_targets,
            target_site="target",
            num_samples=1,
            cutoff=1.0,
        ).accuracy,  # type: ignore
        fix_model_kwargs={
            "vocab_size": train_docs.shape[-1],
            "target_size": train_targets.shape[-1],
            "use_batch_normalization": True,
            "dropout": 0.2,
        },
        var_model_kwargs={
            "num_topics": lambda trial: trial.suggest_int(
                "num_topics", 10, 1000, log=True
            ),
            "hid_size": lambda trial: trial.suggest_int("hid_size", 10, 1000, log=True),
            "hid_num": lambda trial: trial.suggest_int("hid_num", 1, 3),
            "nu_loc": lambda trial: trial.suggest_float("nu_loc", -10, 0),
            "nu_scale": lambda trial: trial.suggest_float("nu_scale", 1, 20),
            "target_scale": lambda trial: trial.suggest_float(
                "target_scale", 1e-2, 10, log=True
            ),
        },
        vectorize_particles=False,
        num_particles=lambda trial: 1,
        gamma=lambda trial: 1.0,
        min_epochs=5,
        max_epochs=lambda trial: 480,
        device=device,
        hooks={
            "svi_self_post_hooks": [
                lambda obj: print(
                    "train metrics",
                    obj.calculate_metrics(
                        train_docs,
                        targets=train_targets,
                        target_site="a",
                        num_samples=25,
                        mean_dim=0,
                        cutoff=None,
                        post_sample_fun=F.sigmoid,
                    ),
                ),
                lambda obj: print(
                    "test metrics",
                    obj.calculate_metrics(
                        test_docs,
                        targets=test_targets,
                        target_site="a",
                        num_samples=25,
                        mean_dim=0,
                        cutoff=None,
                        post_sample_fun=F.sigmoid,
                    ),
                ),
            ]
        },
    )

    study = optuna.create_study(
        study_name="ProdSLDA",
        directions=["maximize"],
        pruner=optuna.pruners.HyperbandPruner(),
        sampler=optuna.samplers.TPESampler(seed=seed),
        storage="sqlite:///prodslda.db",
        load_if_exists=True,
    )
    study.set_user_attr("labeled_data_size", len(train_docs))

    pyro.set_rng_seed(seed)

    study.optimize(objective, n_trials=n_trials)

    print(f"{study.best_trial=}")

    fig = optuna.visualization.plot_param_importances(study)
    fig.show()


def run_optuna_study_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        type=str,
        help="The path to the directory containing the training data",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=25,
    )

    args = parser.parse_args()
    path = Path(args.path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_optuna_study(path=path, n_trials=args.num_trials, device=device)
