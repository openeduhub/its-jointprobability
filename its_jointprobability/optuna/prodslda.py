"""Run hyperparameter optimization for the ProdSLDA model"""
import argparse
import pickle
from collections.abc import Callable
from pathlib import Path
from pprint import pprint
from typing import Any, NamedTuple, Optional

import numpy as np
import optuna
import pyro
import pyro.infer
import torch
from icecream import ic
from its_data.default_pipelines.data import BoW_Data, subset_data_points
from its_jointprobability.data import Split_Data, import_data
from its_jointprobability.optuna.model import set_up_optuna_study
from its_jointprobability.models.prodslda import ProdSLDA, set_up_data, Torch_Data
from its_jointprobability.utils import default_data_loader


class Trial_Spec(NamedTuple):
    name: str
    fix_model_kwargs: dict[str, Any]
    var_model_kwargs: dict[str, Callable[[optuna.Trial], Any]]
    num_particles: Callable[[optuna.Trial], int]
    gamma: Callable[[optuna.Trial], float]
    initial_lr: Callable[[optuna.Trial], float]
    max_epochs: Callable[[optuna.Trial], int]


def run_optuna_study(
    path: Path,
    trial_spec_fun: Callable[[BoW_Data, Torch_Data], Trial_Spec],
    n_trials=25,
    seed: int = 0,
    device: Optional[torch.device] = None,
    train_data_len: Optional[int] = None,
    selected_field: Optional[str] = None,
    verbose=False,
):
    ic.enabled = verbose

    data = import_data(path)
    # only use editorially confirmed data for hyper-parameter tuning
    data = Split_Data(
        subset_data_points(data.train, np.where(data.train.editor_arr)[0]), data.test
    )
    torch_data = set_up_data(data)
    train_docs, train_targets, test_docs, test_targets = torch_data
    trial_spec = trial_spec_fun(data.train, torch_data)

    if train_data_len is not None:
        torch.manual_seed(seed)
        indices = torch.randperm(len(train_docs))[:train_data_len]
        train_docs, train_targets = (
            train_docs[indices],
            {key: value[indices] for key, value in train_targets.items()},
        )

    ic({key: value.sum(-2) for key, value in train_targets.items()})

    def get_eval_fun(
        field: str,
        docs: torch.Tensor,
        targets: torch.Tensor,
        attr: str,
        num_samples: int = 250,
        **kwargs,
    ) -> Callable[[ProdSLDA], float]:
        def fun(model: ProdSLDA) -> float:
            val = getattr(
                ic(
                    model.calculate_metrics(
                        docs,
                        targets=targets,
                        target_site=f"probs_{field}",
                        num_samples=num_samples,
                        mean_dim=0,
                        cutoff=None,
                        **kwargs,
                    )
                ),
                attr,
            )  # type: ignore
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
            batch_size=1000,
        ),
        elbo_choices={"TraceEnum": pyro.infer.TraceEnum_ELBO},
        eval_funs_final=[
            get_eval_fun(
                field,
                test_docs,
                targets,
                "f1_score",
                num_samples=250,
                parallel_sample=False,
            )
            for field, targets in test_targets.items()
            if selected_field is None or field == selected_field
        ],
        eval_fun_prune=lambda model, epoch, loss: get_eval_fun(
            selected_field,
            test_docs,
            test_targets[selected_field],
            "f1_score",  # type: ignore
            num_samples=10,
            parallel_sample=True,
        )(model)
        if selected_field is not None
        else None,
        fix_model_kwargs=trial_spec.fix_model_kwargs,
        var_model_kwargs=trial_spec.var_model_kwargs,
        num_particles=trial_spec.num_particles,
        gamma=trial_spec.gamma,
        max_epochs=trial_spec.max_epochs,
        initial_lr=trial_spec.initial_lr,
        vectorize_particles=False,
        min_epochs=30,
        device=device,
    )

    try:
        with open(f".{trial_spec.name}_sampler.pkl", "rb") as f:
            sampler = pickle.load(f)
    except FileNotFoundError:
        sampler = optuna.samplers.TPESampler(seed=seed)

    study = optuna.create_study(
        study_name=trial_spec.name
        + ("" if selected_field is None else f"_{selected_field}"),
        directions=[
            "maximize"
            for field in train_targets.keys()
            if selected_field is None or field == selected_field
        ],
        pruner=optuna.pruners.HyperbandPruner(),
        sampler=sampler,
        storage="sqlite:///prodslda.db",
        load_if_exists=True,
    )
    study.set_user_attr("labeled_data_size", len(train_docs))

    pyro.set_rng_seed(seed)

    try:
        study.optimize(objective, n_trials=n_trials)
    except (KeyboardInterrupt, RuntimeError):
        pass
    finally:
        with open(f".{trial_spec.name}_sampler.pkl", "wb+") as f:
            pickle.dump(study.sampler, f)

    pprint(study.best_trials)

    fig = optuna.visualization.plot_param_importances(study)
    fig.show()


def default_fix_model_kwargs(
    train_data: BoW_Data, torch_data: Torch_Data
) -> dict[str, Any]:
    return {
        "vocab": train_data.words,
        "id_label_dicts": [
            {
                id: label
                for id, label in zip(
                    train_data.target_data[field].uris,
                    train_data.target_data[field].labels,
                )
            }
            for field in train_data.target_data.keys()
        ],
        "target_names": list(train_data.target_data.keys()),
        "hid_size": 100,
        "hid_num": 1,
        "num_topics": 100,
        "dropout": 0.2,
        "nu_loc": -2.0,
        "nu_scale": 1.5,
        "annealing_factor": 0.5,
        "use_batch_normalization": True,
        "mle_priors": False,
        "correlated_nus": False,
    }


DEFAULT_TRIALS: dict[str, Callable[[BoW_Data, Torch_Data], Trial_Spec]] = {
    "structure": lambda train_data, torch_data: Trial_Spec(
        name="prodslda_encoder-structure",
        fix_model_kwargs=default_fix_model_kwargs(train_data, torch_data),
        var_model_kwargs={
            "hid_size": lambda trial: trial.suggest_int("hid_size", 100, 1000),
            "hid_num": lambda trial: trial.suggest_int("hid_num", 1, 3),
            "num_topics": lambda trial: trial.suggest_int("num_topics", 50, 500),
        },
        num_particles=lambda _: 1,
        gamma=lambda _: 0.5,
        initial_lr=lambda _: 0.1,
        max_epochs=lambda _: 500,
    ),
    "annealing_factor": lambda train_data, torch_data: Trial_Spec(
        name="prodslda_annealing-factor",
        fix_model_kwargs=default_fix_model_kwargs(train_data, torch_data),
        var_model_kwargs={
            "annealing_factor": lambda trial: trial.suggest_float(
                "annealing_factor", 1e-4, 1.0, log=True
            ),
        },
        num_particles=lambda _: 1,
        gamma=lambda _: 0.5,
        initial_lr=lambda _: 0.1,
        max_epochs=lambda _: 500,
    ),
    "priors": lambda train_data, torch_data: Trial_Spec(
        name="prodslda_priors",
        fix_model_kwargs=default_fix_model_kwargs(train_data, torch_data),
        var_model_kwargs={
            "nu_loc": lambda trial: trial.suggest_float("nu_loc", -10.0, 0.0),
            "nu_scale": lambda trial: trial.suggest_float("nu_scale", 0.1, 3, log=True),
        },
        num_particles=lambda _: 1,
        gamma=lambda _: 0.5,
        initial_lr=lambda _: 0.1,
        max_epochs=lambda _: 500,
    ),
    "full": lambda train_data, torch_data: Trial_Spec(
        name="prodslda_full",
        fix_model_kwargs=default_fix_model_kwargs(train_data, torch_data),
        var_model_kwargs={
            "hid_size": lambda trial: trial.suggest_int("hid_size", 50, 1000),
            "hid_num": lambda trial: trial.suggest_int("hid_num", 1, 3),
            "num_topics": lambda trial: trial.suggest_int("num_topics", 25, 500),
            "dropout": lambda trial: trial.suggest_float("dropout", 0.0, 0.9),
            "annealing_factor": lambda trial: trial.suggest_float(
                "annealing_factor", 1e-2, 1.0, log=True
            ),
            "nu_loc": lambda trial: trial.suggest_float("nu_loc", -10.0, 0.0),
            "nu_scale": lambda trial: trial.suggest_float("nu_scale", 0.1, 3, log=True),
            "mle_priors": lambda trial: trial.suggest_categorical(
                "mle_priors", [False, True]
            ),
            "correlated_nus": lambda trial: trial.suggest_categorical(
                "correlated_nus", [False, True]
            ),
        },
        num_particles=lambda trial: 1,
        gamma=lambda trial: trial.suggest_float("gamma", 0.1, 1.0, log=True),
        max_epochs=lambda _: 500,
        initial_lr=lambda trial: trial.suggest_float(
            "initial_lr", 0.001, 0.1, log=True
        ),
    ),
}


def run_optuna_study_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        type=str,
        help="The path to the directory containing the training data",
    )
    parser.add_argument("--trial-type", "-t", type=str, choices=DEFAULT_TRIALS.keys())
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--num-trials", type=int, default=25)
    parser.add_argument("-n", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    path = Path(args.path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_optuna_study(
        path=path,
        trial_spec_fun=DEFAULT_TRIALS[args.trial_type],
        n_trials=args.num_trials,
        train_data_len=args.n,
        selected_field=args.target,
        device=device,
        verbose=args.verbose,
        seed=args.seed,
    )
