import argparse
from pathlib import Path
from typing import Optional

from nlprep import partial

import its_jointprobability.data.disciplines as disciplines
import optuna
import pyro
import pyro.distributions as dist
import pyro.infer
import pyro.optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic
from its_jointprobability.models.model import Model, set_up_optuna_study
from its_jointprobability.utils import Data_Loader, default_data_loader
from pyro.nn.module import to_pyro_module_


class ProdSLDA(Model):
    """
    A modification of the ProdLDA model to support supervized classification.
    """

    return_sites = ("target", "nu", "a")
    return_site_cat_dim = {"nu": 0, "a": -1, "target": -1}

    def __init__(
        self,
        voc_size: int,
        target_size: int,
        num_topics: int,
        layers: int,
        dropout: float,
        nu_loc: float = 0.0,
        nu_scale: float = 10.0,
        target_scale: float = 1.0,
        use_batch_normalization: bool = True,
        observe_negative_targets=torch.tensor(True),
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.vocab_size = voc_size
        self.target_size = target_size
        self.num_topics = num_topics

        self.decoder = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_topics, voc_size),
        )
        if use_batch_normalization:
            self.decoder.append(nn.BatchNorm1d(voc_size, affine=False))
        self.decoder.append(nn.Softmax(-1))
        to_pyro_module_(self.decoder)

        self.encoder = nn.Sequential(
            nn.Linear(voc_size, layers),
            nn.ReLU(),
            nn.Linear(layers, layers),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(layers, num_topics * 2),
        )
        if use_batch_normalization:
            self.encoder.append(nn.BatchNorm1d(num_topics * 2, affine=False))
        to_pyro_module_(self.encoder)

        self.nu_loc = nu_loc
        self.nu_scale = nu_scale
        self.target_scale = target_scale
        self.observe_negative_targets = observe_negative_targets

        self.device = device
        self.to(device)

    def model(self, docs: torch.Tensor, targets: Optional[torch.Tensor] = None):
        targets_plate = pyro.plate("targets", self.target_size, dim=-2)
        docs_plate = pyro.plate("documents", docs.shape[0], dim=-1)

        # # the target application coefficients
        with targets_plate:
            nu = pyro.sample(
                "nu",
                dist.Normal(
                    self.nu_loc * docs.new_ones(self.num_topics),
                    self.nu_scale * docs.new_ones(self.num_topics),
                ).to_event(1),
            )

        with docs_plate:
            logtheta_loc = docs.new_zeros(self.num_topics)
            logtheta_scale = docs.new_ones(self.num_topics)
            logtheta = pyro.sample(
                "logtheta", dist.Normal(logtheta_loc, logtheta_scale).to_event(1)
            )
            if len(logtheta.shape) > 2:
                logtheta = logtheta.squeeze(0)
            theta = F.softmax(logtheta, -1)

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

            with targets_plate:
                a = pyro.sample(
                    "a",
                    dist.Normal(
                        torch.matmul(nu, theta.swapaxes(-1, -2)).squeeze(-2), 10
                    ),
                )
                with pyro.poutine.scale(scale=self.target_scale):
                    target = pyro.sample(
                        "target",
                        dist.Bernoulli(logits=a),  # type: ignore
                        obs=targets.swapaxes(-1, -2) if targets is not None else None,
                        infer={"enumerate": "parallel"},
                    )

    def logtheta_params(self, doc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logtheta_loc, logtheta_logvar = self.encoder(doc).split(self.num_topics, -1)
        logtheta_scale = F.softplus(logtheta_logvar) + 1e-5

        return logtheta_loc, logtheta_scale

    def guide(
        self,
        docs: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ):
        targets_plate = pyro.plate("targets", self.target_size, dim=-2)
        docs_plate = pyro.plate("documents", docs.shape[0], dim=-1)

        # variational parameters for the relationship between topics and targets
        mu_q = pyro.param(
            "mu",
            lambda: torch.randn(self.target_size, self.num_topics)
            .unsqueeze(-2)
            .to(docs.device),
        )
        sigma_q = pyro.param(
            "sigma",
            lambda: docs.new_ones(self.target_size, self.num_topics).unsqueeze(-2),
            constraint=dist.constraints.positive,
        )

        with targets_plate:
            nu_q = pyro.sample("nu", dist.Normal(mu_q, sigma_q).to_event(1))

        with docs_plate:
            logtheta_q = pyro.sample(
                "logtheta", dist.Normal(*self.logtheta_params(docs)).to_event(1)
            )
            theta_q = F.softmax(logtheta_q, -1)

            with targets_plate:
                a_q_scale = pyro.param(
                    "a_q_scale",
                    lambda: docs.new_ones([self.target_size, self.num_topics]),
                    constraint=dist.constraints.positive,
                )
                a_q = pyro.sample(
                    "a",
                    dist.Normal(
                        torch.matmul(nu_q, theta_q.swapaxes(-1, -2)).squeeze(-2),
                        torch.matmul(a_q_scale, theta_q.swapaxes(-1, -2)).squeeze(-2),
                    ),
                )

    def clean_up_posterior_samples(
        self, posterior_samples: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        if "target" in posterior_samples:
            posterior_samples["target"] = posterior_samples["target"].swapaxes(-1, -2)
        if "a" in posterior_samples:
            posterior_samples["a"] = posterior_samples["a"].swapaxes(-1, -2)
        if "nu" in posterior_samples:
            posterior_samples["nu"] = posterior_samples["nu"].squeeze(-2)

        return posterior_samples

    def beta(self):
        # beta matrix elements are the weights of the FC layer on the decoder
        weights = self.decoder.beta.weight.T
        return F.softmax(weights, dim=-1).detach()


def import_model(
    path: Path, device: Optional[torch.device] = None
) -> tuple[ProdSLDA, disciplines.Meta_Data]:
    model: ProdSLDA = torch.load(path / "prodslda", map_location=device)
    # ensure that the model is correctly running on the given device
    model.device = device

    metadata = disciplines.get_metadata(path)
    return model, metadata


def train_model(
    data_loader: Data_Loader,
    voc_size: int,
    target_size: int,
    num_topics: int = 100,  # best: 50
    layers: int = 100,  # best: 440
    dropout: float = 0.2,
    nu_loc: float = -4.8,  # best: -6.0
    nu_scale: float = 8.6,  # best: 18.2
    min_epochs: int = 100,
    max_epochs: int = 250,
    target_scale: float = 1.5,
    initial_lr: float = 0.1,  # best: 0.082
    gamma: float = 1.0,
    device: Optional[torch.device] = None,
    seed: int = 0,
) -> ProdSLDA:
    pyro.set_rng_seed(seed)

    prodslda = ProdSLDA(
        voc_size=voc_size,
        target_size=target_size,
        num_topics=num_topics,
        layers=layers,
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

    ic(prodslda.training)

    return prodslda.eval()


def run_optuna_study(
    path: Path,
    n: int = 100,
    n_trials=25,
    seed: int = 0,
    device: Optional[torch.device] = None,
):
    train_data = disciplines.get_train_data(path, n=n, always_include_confirmed=False)

    train_docs: torch.Tensor = train_data.docs
    train_targets: torch.Tensor = train_data.targets

    ic(train_docs.shape)
    ic(train_targets.shape)
    ic(train_targets.sum(-2))

    test_data = disciplines.get_test_data(path)
    test_docs = test_data.docs
    test_targets = test_data.targets

    ic(test_docs.shape)
    ic(test_targets.shape)

    objective = set_up_optuna_study(
        factory=ProdSLDA,
        data_loader=default_data_loader(
            train_docs,
            train_targets,
            device=device,
            dtype=torch.float,
        ),
        elbo_choices={
            "TraceGraph": pyro.infer.TraceGraph_ELBO,
        },
        eval_fun_final=lambda obj: obj.calculate_metrics(
            test_docs,
            targets=test_targets,
            target_site="a",
            num_samples=1000,
            mean_dim=0,
            cutoff=None,
            post_sample_fun=F.sigmoid,
        ).f1_score,  # type: ignore
        eval_fun_prune=lambda obj, step, loss: obj.calculate_metrics(
            test_docs,
            targets=test_targets,
            target_site="a",
            num_samples=10,
            mean_dim=0,
            cutoff=None,
            post_sample_fun=F.sigmoid,
        ).f1_score,  # type: ignore
        eval_freq=5,
        fix_model_kwargs={
            "voc_size": train_docs.shape[-1],
            "target_size": train_targets.shape[-1],
            "use_batch_normalization": True,
            "dropout": 0.2,
        },
        var_model_kwargs={
            "num_topics": lambda trial: trial.suggest_int("num_topics", 50, 500, 10),
            "layers": lambda trial: trial.suggest_int("layers", 50, 500, 10),
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
                        target_site="target",
                        num_samples=100,
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
                        target_site="target",
                        num_samples=100,
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
        direction="maximize",
        pruner=optuna.pruners.HyperbandPruner(),
        sampler=optuna.samplers.TPESampler(seed=seed),
        storage="sqlite:///prodslda.db",
        load_if_exists=True,
    )
    study.set_user_attr("labeled_data_size", len(train_docs))

    pyro.set_rng_seed(seed)

    study.optimize(objective, n_trials=n_trials)

    print(f"{study.best_value=}")
    print(f"{study.best_params=}")
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
        "-n",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=25,
    )

    args = parser.parse_args()

    path = Path(args.path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_optuna_study(path=path, n=args.n, n_trials=args.num_trials, device=device)
