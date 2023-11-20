from typing import Optional
from collections import deque
from collections.abc import Callable, Collection, Iterator

import numpy as np
import pyro.infer
import pyro.optim

from tqdm import trange


def run_svi(
    model: Callable,
    guide: Callable,
    elbo,
    batch_strategy: Optional[Iterator[tuple[bool, Collection[int]]]] = None,
    batches_per_epoch: int = 1,
    initial_lr: float = 0.001,
    gamma: float = 0.1,
    betas: tuple[float, float] = (0.95, 0.999),
    max_epochs: int = 1000,
    min_epochs: int = 100,
    min_rel_std: float = 0.1,
    z_score_num: int = 10,
    min_z_score: float = 1.0,
    logger=None,
    savefile=None,
    **kwargs,
) -> list[float]:
    if savefile is not None:
        try:
            pyro.get_param_store().load(savefile)
            return list()
        except FileNotFoundError:
            pass
    optim = pyro.optim.ClippedAdam(
        {
            # initial learning rate
            "lr": initial_lr,
            # final learning rate will be gamma * initial_lr
            "lrd": gamma ** (1 / (max_epochs * batches_per_epoch)),
            # hyperparameters for the per-parameter momentum
            "betas": betas,
        }
    )
    svi = pyro.infer.SVI(model, guide, optim, elbo)

    not logger or logger.info("----------------------")
    not logger or logger.info("starting new inference")

    losses = list()
    # collect the last z-scores using a ring buffer
    z_scores = deque(maxlen=z_score_num)
    for epoch in trange(max_epochs):
        if batch_strategy:
            batch_losses = list()
            for last_batch_in_epoch, minibatch in batch_strategy:
                # if a batch size has been given,
                # assume that the model and guide
                # are parameterized by a subsample index list
                batch_losses.append(svi.step(batch=minibatch, **kwargs))
                # break if this was the last batch
                if last_batch_in_epoch:
                    break

            loss = np.mean(batch_losses)
        else:
            loss = svi.step(**kwargs)

        losses.append(loss)

        # compute the last z-score
        mean = np.mean(losses[-25:])
        std = np.std(losses[-25:])
        rel_std = np.abs(std / mean)
        z_scores.append(np.abs((losses[-1] - mean) / std))
        if (
            epoch > min_epochs
            and all(z_score < min_z_score for z_score in z_scores)
            and rel_std < min_rel_std
        ):
            not logger or logger.info(f"breaking after {epoch=}")
            break

        # log every few epochs
        if epoch % 25 == 0:
            not logger or logger.info(
                f"{epoch=}\t{loss=:.3g}\t{rel_std=:.3f}\t{z_scores[-1]=:.2f}"
            )

    print("done!")
    if savefile is not None:
        pyro.get_param_store().save(savefile)
    return losses
