"""The webservice that allows for interaction with the Bayesian model."""
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import torch
from fastapi import APIRouter, FastAPI
from pydantic import BaseModel, Field

from its_jointprobability._version import __version__
from its_jointprobability.data import load_model
from its_jointprobability.models.model import Prediction_Score
from its_jointprobability.models.prodslda import ProdSLDA

log = logging.getLogger(__name__)


DEBUG = False
MODEL_DIR = None


model: Optional[ProdSLDA] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    log.info("Loading models")
        
    # import the model and auxiliary data
    # if the model location has not been set, try to look it up in the
    # environment variables
    model_dir = os.environ.get("DATA_DIR", MODEL_DIR)
    if model_dir is None:
        raise RuntimeError("Error: model directory not set.")
        #print("Error: model directory not set.")
        # return
    # find a suitable backend
    devices = ['cuda', 'mps', 'cpu']
    for d in devices:
        module = getattr(torch, d, None)
        if module and module.is_available():
            device = torch.device(d)
            break
    else:
        raise RuntimeError(f"No device found (tried: {', '.join(devices)})")
    
    log.info("Using device %s", device)
    model = load_model(ProdSLDA, Path(model_dir), device=device)

    # initialize the baseline distributions by calling prediction on a dummy
    # text.
    # TODO: move this to post model training and save the baseline
    # distributions
    list(
        model.predict_from_texts(
            model.vocab[0], model.vocab[1], tokens=model.vocab, num_samples=2
        )
    )


    yield

    log.info("Unloading models")
    model = None


router = APIRouter(lifespan=lifespan)

class Prediction_Data(BaseModel):
    """Input to be used for prediction."""

    text: str
    num_samples: int = Field(
        default=500 if not DEBUG else 2, gt=1, le=100000 if not DEBUG else 10
    )
    num_predictions: int = Field(default=10, gt=0)
    interval_size: float = Field(default=0.8, gt=0.0, lt=1.0)

class Prediction_Result(BaseModel):
    """The output of the prediction."""

    predictions: dict[str, list[Prediction_Score]]
    version: str = __version__

@router.post(
    "/bayesian-predictions",
    summary="Predict the metadata fitting the given text.",
    description="""
    Note that all categories are not filtered out. Instead, they
    are sorted by their mean predicted probability of being
    relevant to the text.
    
    Parameters
    ----------
    text : str
        The text to be analyzed.
    num_samples : int
        The number of samples to use in order to estimate the fit
        of each discipline. Higher numbers will result in less
        variance between calls, but take more time.
    num_predictions : int
        The number of predicted disciplines (sorted by relevance)
        to return. This does not affect performance; it simply
        serves as an initial filtering tool.
    interval_size : float (0, 1]
        The size of the credibility interval for the probability
        that a discipline is assigned to the given text.
        E.g. at 0.8, there is a probability of 80% that the
        predicted probability of the discipline belonging to the
        text is within the returned interval.

    Returns
    -------
    predictions : dict[str, list[Prediction]]
        Map from predicted metadatum to predictions for this
        metadatum.
    version : str
        The version of the prediction tool.

    Prediction
    ----------
    id : str
        The URI of the category.
    name : str
        The label of the category.
    mean_prob : float [0, 1]
        The mean of the predicted probabilities that this category
        belongs to the given text.
    median_prob : float [0, 1]
        The median of the above probabilities.
    prob_interval : 2-tuple of floats in [0, 1]
        The credibility interval of the predicted probabilities
        above.
    """,
)
def predict(inp: Prediction_Data) -> Prediction_Result:
    # , model: Annotated[ProdSLDA, get_model], tokens: Annotated[Sequence[str], get_tokens]
    assert model
    tokens = model.vocab
    predictions = next(
        model.predict_from_texts(
            inp.text,
            tokens=tokens,
            num_samples=inp.num_samples,
            interval_size=inp.interval_size,
        )
    )
    # sort the predictions and only keep the most relevant
    predictions = {
        key: sorted(
            value,
            key=lambda x: x.mean_prob,
            reverse=True,
        )[: min(len(value), inp.num_predictions)]
        for key, value in predictions.items()
    }
    return Prediction_Result(predictions=predictions)

