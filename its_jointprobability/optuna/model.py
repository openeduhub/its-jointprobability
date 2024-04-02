from collections.abc import Callable, Collection, Iterable, Mapping
from functools import partial
from pprint import pprint
from typing import Any, Optional, TypeVar

import optuna
import pandas as pd
import pyro
import pyro.infer
import torch
from icecream import ic
from its_jointprobability.models.model import Model
from its_jointprobability.utils import Data_Loader, Quality_Result, quality_measures

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
    betas: Optional[Callable[[optuna.trial.Trial], tuple[float, float]]] = None,
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
            "max_epoches", min_epochs, min_epochs + 50 * 6, step=50
        )
    if betas is None:
        betas = lambda trial: (
            trial.suggest_float("beta_1", 0.01, 0.99, log=True),
            trial.suggest_float("beta_2", 0.01, 0.999, log=True),
        )

    def objective(trial: optuna.trial.Trial) -> list[float]:
        trial_kwargs = {key: fun(trial) for key, fun in var_model_kwargs.items()}
        initial_lr_trial = initial_lr(trial)
        gamma_trial = gamma(trial)
        num_particles_trial = num_particles(trial)
        trial_elbo_num = trial.suggest_categorical(
            "elbo_num", list(elbo_choices.keys())
        )
        trial_betas = betas(trial)
        # adjust the maximum number of epochs w.r.t. the number of particles
        # to keep training time roughly constant
        max_epochs_trial = int(max_epochs(trial) / num_particles_trial)

        pyro.get_param_store().clear()
        # prevent the experiment from studying different
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
            betas=trial_betas,
        )

        return [fun(model) for fun in eval_funs_final]

    return objective
