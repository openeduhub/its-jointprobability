import argparse
import math
import operator as op
from collections.abc import Callable, Collection, Sequence
from pathlib import Path
from typing import NamedTuple, Optional
from pprint import pprint

import numpy as np
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
        correlated_nus: bool = True,
        bias_from_previous_targets: bool = True,
        device: Optional[torch.device] = None,
    ):
        vocab_size = len(vocab)
        target_sizes = [len(id_label_dict) for id_label_dict in id_label_dicts]

        ns = list(range(len(target_sizes)))

        self.return_sites = tuple(
            [f"target_{i}" for i in ns]
            + [f"probs_{i}" for i in ns]
            + [f"sigma_{i}" for i in ns[1:]]
            + ["nu", "a"]
        )
        self.return_site_cat_dim = (
            {f"target_{i}": -2 for i in ns}
            | {f"probs_{i}": -2 for i in ns}
            | {f"sigma_{i}": -4 for i in ns[1:]}
            | {"nu": -4, "a": -2}
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
        self.correlated_nus = correlated_nus
        self.bias_from_previous_targets = bias_from_previous_targets
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
            "correlated_nus": self.correlated_nus,
            "bias_from_previous_targets": self.bias_from_previous_targets,
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

        cur_hid_size, prev_hid_size = hid_size // 2, hid_size
        for _ in range(hid_num - 1):
            self.encoder.append(nn.Tanh())
            self.encoder.append(nn.Linear(prev_hid_size, cur_hid_size))
            prev_hid_size = cur_hid_size
            cur_hid_size = prev_hid_size // 2

        self.encoder.append(nn.Tanh())
        self.encoder.append(nn.Dropout(dropout))
        self.encoder.append(nn.Linear(prev_hid_size, num_topics * 2))
        if use_batch_normalization:
            self.encoder.append(nn.BatchNorm1d(num_topics * 2, affine=False))

        self.to(device)
        ic(self)

    def model(
        self,
        docs: torch.Tensor,
        *targets: torch.Tensor,
        obs_masks: Optional[Sequence[torch.Tensor]] = None,
    ):
        n = sum(self.target_sizes)
        # if no observations mask has been given, ignore any docs that
        # do not have any assigned labels for that given target
        # or any non-assigned labels (depending on the model's settings)
        if obs_masks is None:
            obs_masks = self._get_obs_mask(*targets)

        docs_plate = pyro.plate("documents_plate", docs.shape[0], dim=-1)

        # the matrix mapping the relationship between latent topic and targets
        nu = pyro.sample(
            "nu",
            dist.Normal(
                torch.tensor(self.nu_loc, device=docs.device),
                torch.tensor(self.nu_scale, device=docs.device),
            )
            .expand(torch.Size([self.num_topics, n]))
            .to_event(2),
        )
        ic(nu.shape)

        # the influence of the previous targets on subsequent ones
        if self.bias_from_previous_targets:
            sigmas: list[torch.Tensor] = [torch.tensor([])]
            for i, (cum_size, cur_size) in enumerate(
                zip(np.cumsum(self.target_sizes), self.target_sizes[1:])
            ):
                sigma = pyro.sample(
                    f"sigma_{i+1}",
                    dist.Normal(docs.new_zeros([cum_size, cur_size]), 1).to_event(2),
                )
                ic(sigma.shape)
                sigmas.append(sigma)

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

            prev_targets: list[torch.Tensor] = list()
            with pyro.poutine.scale(scale=self.target_scale):
                for i in range(len(self.target_sizes)):
                    # the influence of previous target values on the current one
                    if self.bias_from_previous_targets and i > 0:
                        prev_targets_tensor = torch.concat(prev_targets, dim=-1)
                        prev_targets_effect = torch.matmul(
                            prev_targets_tensor,
                            sigmas[i],  # type: ignore
                        )
                        ic(prev_targets_effect.shape)
                    else:
                        prev_targets_effect = torch.tensor(0)

                    probs = pyro.deterministic(
                        f"probs_{i}", F.sigmoid(a_split[i] + prev_targets_effect)
                    )
                    ic(probs.shape)

                    target = pyro.sample(
                        f"target_{i}",
                        dist.Bernoulli(probs).to_event(1),  # type: ignore
                        obs=targets[i] if len(targets) > i else None,
                        obs_mask=obs_masks[i] if len(obs_masks) > i else None,
                    )
                    ic(target.shape)

                    prev_targets.append(target)

    def guide(
        self,
        docs: torch.Tensor,
        *targets: torch.Tensor,
        obs_masks: Optional[Sequence[torch.Tensor]] = None,
    ):
        n = sum(self.target_sizes)
        if obs_masks is None:
            obs_masks = self._get_obs_mask(*targets)

        docs_plate = pyro.plate("documents_plate", docs.shape[0], dim=-1)
        ic(docs.shape)

        # variational parameters for the relationship between topics and targets
        mu_q = pyro.param(
            "mu", lambda: torch.randn(self.num_topics, n, device=docs.device)
        )
        cov_diag = pyro.param(
            "cov_diag",
            lambda: docs.new_ones(self.num_topics, n),
            constraint=dist.constraints.positive,
        )
        if self.correlated_nus:
            cov_factor = pyro.param(
                "cov_factor",
                lambda: docs.new_ones(self.num_topics, n, self.cov_rank),
                constraint=dist.constraints.positive,
            )

            nu_q = pyro.sample(
                "nu",
                dist.LowRankMultivariateNormal(mu_q, cov_factor, cov_diag).to_event(1),
            )
        else:
            nu_q = pyro.sample(
                "nu",
                dist.Normal(mu_q, cov_diag).to_event(2),
            )

        ic(nu_q.shape)

        if self.bias_from_previous_targets:
            sigma_qs: list[torch.Tensor] = [torch.tensor([])]
            for i, (cum_size, cur_size) in enumerate(
                zip(np.cumsum(self.target_sizes), self.target_sizes[1:])
            ):
                sigma_loc = pyro.param(
                    f"sigma_{i+1}_loc", docs.new_zeros([cum_size, cur_size])
                )
                sigma_scale = pyro.param(
                    f"sigma_{i+1}_scale",
                    docs.new_ones([cum_size, cur_size]),
                    constraint=dist.constraints.positive,
                )
                sigma_q = pyro.sample(
                    f"sigma_{i+1}", dist.Normal(sigma_loc, sigma_scale).to_event(2)
                )
                ic(sigma_q.shape)
                sigma_qs.append(sigma_q)

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

            prev_targets: list[torch.Tensor] = list()
            with pyro.poutine.scale(scale=self.target_scale):
                for i in range(len(self.target_sizes)):
                    # the influence of previous target values on the current one
                    if self.bias_from_previous_targets and i > 0:
                        prev_targets_tensor = torch.concat(prev_targets, dim=-1)
                        prev_targets_effect = torch.matmul(
                            prev_targets_tensor,
                            sigma_qs[i],  # type: ignore
                        )
                    else:
                        prev_targets_effect = torch.tensor(0)

                    probs = F.sigmoid(a_q_split[i] + prev_targets_effect)

                    target_q = pyro.sample(
                        f"target_{i}_unobserved",
                        dist.Bernoulli(probs).to_event(1),  # type: ignore
                    )
                    prev_targets.append(target_q)

    def logtheta_params(self, doc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pyro.module("encoder", self.encoder, update_module_params=True)
        logtheta_loc, logtheta_logvar = self.encoder(doc).split(self.num_topics, -1)
        logtheta_scale = F.softplus(logtheta_logvar) + 1e-7

        return logtheta_loc, logtheta_scale

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

    def _get_obs_mask(self, *targets: torch.Tensor) -> list[torch.Tensor]:
        return [target.sum(-1) > 0 for target in targets]


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
    num_topics: int = 100,
    hid_size: int = 200,
    hid_num: int = 1,
    dropout: float = 0.2,
    nu_loc: float = -1.7,
    nu_scale: float = 15,
    min_epochs: int = 100,
    max_epochs: int = 500,
    target_scale: float = 1.0,
    initial_lr: float = 0.1,
    gamma: float = 0.75,
    seed: int = 0,
    device: Optional[torch.device] = None,
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

    titles = {key: data.train.target_data[key].labels for key in train_targets.keys()}
    eval_sites = {key: f"probs_{i}" for i, key in enumerate(train_targets.keys())}

    # evaluate the newly trained model
    print("evaluating model on train data")
    eval_model(
        model=prodslda,
        data=train_docs,
        targets=train_targets,
        target_values=titles,
        eval_sites=eval_sites,
    )

    if len(test_docs) > 0:
        print("evaluating model on test data")
        eval_model(
            model=prodslda,
            data=test_docs,
            targets=test_targets,
            target_values=titles,
            eval_sites=eval_sites,
        )


def run_optuna_study(
    path: Path,
    n_trials=25,
    seed: int = 0,
    device: Optional[torch.device] = None,
    train_data_len: Optional[int] = None,
    verbose=False,
):
    data = load_data(path)
    train_docs, train_targets, test_docs, test_targets = set_up_data(data)

    if train_data_len is not None:
        torch.manual_seed(seed)
        indices = torch.randperm(len(train_docs))[:train_data_len]
        train_docs, train_targets = (
            train_docs[indices],
            {key: value[indices] for key, value in train_targets.items()},
        )

    ic({key: value.sum(-2) for key, value in train_targets.items()})

    ic.enabled = verbose

    def get_eval_fun(
        i: int, docs: torch.Tensor, targets: torch.Tensor
    ) -> Callable[[ProdSLDA], float]:
        def fun(model: ProdSLDA) -> float:
            val = model.calculate_metrics(
                docs,
                targets=targets,
                target_site=f"probs_{i}",
                num_samples=100,
                mean_dim=0,
                cutoff=0.2,
            ).accuracy
            assert isinstance(val, float)
            return val

        return fun

    objective = set_up_optuna_study(
        factory=ProdSLDA,
        data_loader=default_data_loader(
            train_docs,
            *train_targets.values(),
            device=device,
            dtype=torch.float,
        ),
        elbo_choices={"TraceGraph": pyro.infer.TraceGraph_ELBO},
        eval_funs_final=[
            get_eval_fun(i, test_docs, targets)
            for i, targets in enumerate(test_targets.values())
        ],
        fix_model_kwargs={
            "vocab": data.train.words,
            "id_label_dicts": [
                {
                    id: label
                    for id, label in zip(
                        data.train.target_data[field].uris,
                        data.train.target_data[field].labels,
                    )
                }
                for field in train_targets.keys()
            ],
            "target_names": list(train_targets.keys()),
            "use_batch_normalization": True,
            "dropout": 0.2,
            "target_scale": 1,
            "nu_loc": -2,
            "nu_scale": 10,
            "num_topics": 200,
        },
        var_model_kwargs={
            "hid_size": lambda trial: trial.suggest_int("hid_size", 100, 500, log=True),
            "hid_num": lambda trial: trial.suggest_int("hid_num", 1, 3),
            "correlated_nus": lambda trial: trial.suggest_categorical(
                "correlated_nus", [False, True]
            ),
            "bias_from_previous_targets": lambda trial: trial.suggest_categorical(
                "bias_from_previous_targets", [False, True]
            ),
        },
        vectorize_particles=False,
        # num_particles=lambda trial: 3,
        gamma=lambda trial: 1.0,
        min_epochs=5,
        max_epochs=lambda trial: 100,
        initial_lr=lambda trial: trial.suggest_float(
            "initial_lr", 1e-3, 1e-1, log=True
        ),
        device=device,
    )

    study = optuna.create_study(
        study_name="ProdSLDA",
        directions=["maximize" for _ in train_targets],
        pruner=optuna.pruners.HyperbandPruner(),
        sampler=optuna.samplers.TPESampler(seed=seed),
        storage="sqlite:///prodslda.db",
        load_if_exists=True,
    )
    study.set_user_attr("labeled_data_size", len(train_docs))

    pyro.set_rng_seed(seed)

    study.optimize(objective, n_trials=n_trials)

    pprint(study.best_trial)

    fig = optuna.visualization.plot_param_importances(study)
    fig.show()


def run_optuna_study_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        type=str,
        help="The path to the directory containing the training data",
    )
    parser.add_argument("--num-trials", type=int, default=25)
    parser.add_argument("-n", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    path = Path(args.path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_optuna_study(
        path=path,
        n_trials=args.num_trials,
        train_data_len=args.n,
        device=device,
        verbose=args.verbose,
    )
