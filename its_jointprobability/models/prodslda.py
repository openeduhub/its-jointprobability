import argparse
import math
import operator as op
from collections.abc import Callable, Collection, Sequence
from pathlib import Path
from typing import NamedTuple, Optional
from pprint import pprint

import pickle
from data_utils.default_pipelines.data import BoW_Data, subset_data_points
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
from its_jointprobability.models.model import (
    Model,
    Simple_Model,
    eval_model,
    set_up_optuna_study,
)
from its_jointprobability.utils import Data_Loader, default_data_loader
from data_utils.defaults import Fields


class ProdSLDA(Model):
    """
    A modification of the ProdLDA model to support supervised classification.
    """

    def __init__(
        self,
        vocab: Sequence[str],
        id_label_dicts: Collection[dict[str, str]],
        target_names: Collection[str],
        num_topics: int = 320,
        cov_rank: Optional[int] = None,
        hid_size: int = 350,
        hid_num: int = 1,
        dropout: float = 0.2,
        nu_loc: float = -7.5,
        nu_scale: float = 0.6,
        use_batch_normalization: bool = True,
        correlated_nus: bool = False,
        mle_priors: bool = False,
        device: Optional[torch.device] = None,
    ):
        # save the given arguments so that they can be exported later
        self.args = locals().copy()
        del self.args["self"]
        del self.args["device"]

        # dynamically set the return sites
        vocab_size = len(vocab)
        target_sizes = [len(id_label_dict) for id_label_dict in id_label_dicts]
        ns = list(range(len(target_sizes)))

        self.return_sites = tuple(
            [f"target_{i}" for i in ns] + [f"probs_{i}" for i in ns] + ["nu", "a"]
        )
        self.return_site_cat_dim = (
            {f"target_{i}": -2 for i in ns}
            | {f"probs_{i}": -2 for i in ns}
            | {"nu": -4, "a": -2}
        )

        super().__init__()

        # because we tend to have significantly more words than targets,
        # the topic modeling part of the model can dominate the classification.
        # to combat this, we scale the loss function of the latter accordingly.
        self.target_scale = max(
            1.0, np.log(vocab_size * num_topics / sum(target_sizes))
        )

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
        self.nu_loc = float(nu_loc)
        self.nu_scale = float(nu_scale)
        self.use_batch_normalization = use_batch_normalization
        self.correlated_nus = correlated_nus
        self.mle_priors = mle_priors
        self.device = device

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
        *targets: torch.Tensor | None,
        obs_masks: Optional[Sequence[torch.Tensor | None]] = None,
    ):
        n = sum(self.target_sizes)
        # if no observations mask has been given, ignore any docs that
        # do not have any assigned labels for that given target
        # or any non-assigned labels (depending on the model's settings)
        if obs_masks is None:
            obs_masks = self._get_obs_mask(*targets)

        docs_plate = pyro.plate("documents_plate", docs.shape[0], dim=-1)

        # the matrix mapping the relationship between latent topic and targets
        nu_loc_fun = lambda: self.nu_loc * docs.new_ones([self.num_topics, n])
        nu_scale_fun = lambda: self.nu_scale * docs.new_ones([self.num_topics, n])
        nu_loc = pyro.param("nu_loc", nu_loc_fun) if self.mle_priors else nu_loc_fun()
        nu_scale = (
            pyro.param("nu_scale", nu_scale_fun, constraint=dist.constraints.positive)
            if self.mle_priors
            else nu_scale_fun()
        )

        nu = pyro.sample("nu", dist.Normal(nu_loc, nu_scale).to_event(2))
        ic(nu.shape)

        with docs_plate:
            logtheta_loc_fun = lambda: docs.new_zeros(self.num_topics)
            logtheta_scale_fun = lambda: docs.new_ones(self.num_topics)
            logtheta_loc = (
                pyro.param("logtheta_loc", logtheta_loc_fun)
                if self.mle_priors
                else logtheta_loc_fun()
            )
            logtheta_scale = (
                pyro.param(
                    "logtheta_scale",
                    logtheta_scale_fun,
                    constraint=dist.constraints.positive,
                )
                if self.mle_priors
                else logtheta_scale_fun()
            )

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

            probs_col = [
                pyro.deterministic(f"probs_{i}", F.sigmoid(a_local))
                for i, a_local in enumerate(a.split(self.target_sizes, -1))
            ]

            with pyro.poutine.scale(scale=self.target_scale):
                for i, probs in enumerate(probs_col):
                    ic(probs.shape)
                    with pyro.plate(f"target_{i}_plate", probs.shape[-1]):
                        targets_i = targets[i] if len(targets) > i else None
                        obs_masks_i = obs_masks[i] if len(obs_masks) > i else None
                        target = pyro.sample(
                            f"target_{i}",
                            dist.Bernoulli(probs.swapaxes(-1, -2)),  # type: ignore
                            obs=targets_i.swapaxes(-1, -2)
                            if targets_i is not None
                            else None,
                            obs_mask=obs_masks_i,
                            infer={"enumerate": "parallel"},
                        ).swapaxes(-1, -2)
                        ic(target.shape)

    def guide(
        self,
        docs: torch.Tensor,
        *targets: torch.Tensor | None,
        obs_masks: Optional[Sequence[torch.Tensor | None]] = None,
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
            cov_factor = (
                pyro.param(
                    "cov_factor",
                    lambda: docs.new_ones(self.num_topics, n, self.cov_rank),
                    constraint=dist.constraints.positive,
                )
                + 1e-7
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

            probs_col = [
                F.sigmoid(a_local)
                for i, a_local in enumerate(a_q.split(self.target_sizes, -1))
            ]

            with pyro.poutine.scale(scale=self.target_scale):
                for i in range(len(self.target_sizes)):
                    probs = probs_col[i]
                    with pyro.plate(f"target_{i}_plate"):
                        target_q = pyro.sample(
                            f"target_{i}_unobserved",
                            dist.Bernoulli(probs.swapaxes(-1, -2)),  # type: ignore
                            infer={"enumerate": "parallel"},
                        ).swapaxes(-1, -2)

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

    def _get_obs_mask(self, *targets: torch.Tensor | None) -> list[torch.Tensor | None]:
        return [
            target.sum(-1) > 0 if target is not None else None for target in targets
        ]


class Torch_Data(NamedTuple):
    train_docs: torch.Tensor
    train_targets: dict[str, torch.Tensor]
    test_docs: torch.Tensor
    test_targets: dict[str, torch.Tensor]


def set_up_data(data: Split_Data) -> Torch_Data:
    def to_tensor(data: BoW_Data) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # use the from_numpy function, as this way, the two share memory
        docs: torch.Tensor = torch.from_numpy(data.bows)
        targets = {
            key: torch.from_numpy(value.arr) for key, value in data.target_data.items()
        }

        ic(docs.shape)
        ic({field: target.shape for field, target in targets.items()})
        ic({field: train_target.sum(-2) for field, train_target in targets.items()})

        return docs, targets

    return Torch_Data(*to_tensor(data.train), *to_tensor(data.test))


def train_model(
    data_loader: Data_Loader,
    vocab: Sequence[str],
    id_label_dicts: Collection[dict[str, str]],
    target_names: Collection[str],
    min_epochs: int = 10,
    max_epochs: int = 500,
    num_particles: int = 3,
    initial_lr: float = 0.1,
    gamma: float = 0.2,
    seed: int = 0,
    device: Optional[torch.device] = None,
    **kwargs,
) -> ProdSLDA:
    pyro.set_rng_seed(seed)

    prodslda = ProdSLDA(
        vocab=vocab,
        id_label_dicts=id_label_dicts,
        target_names=target_names,
        device=device,
        **kwargs,
    )

    prodslda.run_svi(
        data_loader=data_loader,
        elbo=pyro.infer.TraceEnum_ELBO(
            num_particles=num_particles,
            max_plate_nesting=2,
            vectorize_particles=False,
        ),
        min_epochs=min_epochs,
        max_epochs=max_epochs // num_particles,
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
        default=500,
        help="The maximum number of training epochs per batch of data",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=None,
        help="The maximum number of training documents",
    )
    parser.add_argument(
        "--only-confirmed",
        action="store_true",
        help="Whether to only include editorially confirmed materials for training.",
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

    if args.only_confirmed:
        train_data = subset_data_points(data.train, np.where(data.train.editor_arr)[0])
        data = Split_Data(train_data, data.test)

    train_docs, train_targets, test_docs, test_targets = set_up_data(data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_loader = default_data_loader(
        train_docs,
        *train_targets.values(),
        device=device,
        dtype=torch.float,
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

    save_model(prodslda, path, "_".join(train_targets.keys()))
    eval_sites = {key: f"probs_{i}" for i, key in enumerate(train_targets.keys())}

    run_evaluation(prodslda, data, eval_sites)


def run_evaluation(model: Simple_Model, data: Split_Data, eval_sites: dict[str, str]):
    train_docs, train_targets, test_docs, test_targets = set_up_data(data)
    titles = {key: value.labels for key, value in data.train.target_data.items()}

    # evaluate the newly trained model
    print()
    print("------------------------------")
    print("evaluating model on train data")
    eval_model(
        model,
        train_docs,
        targets=train_targets,
        target_values=titles,
        eval_sites=eval_sites,
    )

    if len(test_docs) > 0:
        print()
        print("-----------------------------")
        print("evaluating model on test data")
        eval_model(
            model,
            test_docs,
            targets=test_targets,
            target_values=titles,
            eval_sites=eval_sites,
        )

        print()
        print("-----------------------------------------------------------")
        print("evaluating model on test data, providing all other metadata")
        for index, key in enumerate(titles.keys()):
            targets_without_current = [test_docs] + [
                targets if i != index else None
                for i, targets in enumerate(test_targets.values())
            ]
            eval_model(
                model,
                *targets_without_current,
                targets={key: test_targets[key]},
                target_values={key: titles[key]},
                eval_sites={key: eval_sites[key]},
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
                cutoff=None,
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
            batch_size=4000,
        ),
        elbo_choices={"TraceGraph": pyro.infer.TraceEnum_ELBO},
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
            # defaults; overridden if also present in the variable kwargs
            "hid_size": 350,
            "hid_num": 1,
            "num_topics": 320,
            "dropout": 0.2,
            "nu_loc": -2,
            "nu_scale": 5,
            "use_batch_normalization": True,
            "mle_priors": False,
            "correlated_nus": False,
        },
        var_model_kwargs={
            "hid_size": lambda trial: trial.suggest_int("hid_size", 50, 500),
            "hid_num": lambda trial: trial.suggest_int("hid_num", 1, 3),
            "num_topics": lambda trial: trial.suggest_int("num_topics", 50, 500),
            "dropout": lambda trial: trial.suggest_float("dropout", 0.0, 0.9),
            "nu_loc": lambda trial: trial.suggest_float("nu_loc", -10.0, 0.0),
            "nu_scale": lambda trial: trial.suggest_float("nu_scale", 0.1, 3, log=True),
            # "mle_priors": lambda trial: trial.suggest_categorical(
            #     "mle_priors", [True, False]
            # ),
            # "correlated_nus": lambda trial: trial.suggest_categorical(
            #     "correlated_nus", [True, False]
            # ),
        },
        vectorize_particles=False,
        num_particles=lambda trial: trial.suggest_int("num_particles", 1, 7),
        gamma=lambda trial: trial.suggest_float("gamma", 0.1, 1.0, log=True),
        min_epochs=5,
        max_epochs=lambda trial: 500,
        initial_lr=lambda trial: trial.suggest_float("initial_lr", 0.01, 1.0, log=True),
        device=device,
    )

    NAME = "prodslda_all-f1"

    try:
        with open(f".{NAME}_sampler.pkl", "rb") as f:
            sampler = pickle.load(f)
    except FileNotFoundError:
        sampler = optuna.samplers.TPESampler(seed=seed)

    study = optuna.create_study(
        study_name=NAME,
        directions=["maximize" for _ in train_targets],
        pruner=optuna.pruners.HyperbandPruner(),
        sampler=sampler,
        storage=f"sqlite:///prodslda.db",
        load_if_exists=True,
    )
    study.set_user_attr("labeled_data_size", len(train_docs))

    pyro.set_rng_seed(seed)

    try:
        study.optimize(objective, n_trials=n_trials)
    except (KeyboardInterrupt, RuntimeError):
        pass
    finally:
        with open(".prodslda_sampler.pkl", "wb+") as f:
            pickle.dump(study.sampler, f)

    pprint(study.best_trials)

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
    parser.add_argument("--seed", type=int, default=0)
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
        seed=args.seed,
    )
